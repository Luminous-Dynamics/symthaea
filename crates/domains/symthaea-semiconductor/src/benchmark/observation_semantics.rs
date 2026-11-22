// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed sensor observation states and explicit cross-schema correspondence.
//!
//! Missingness is not a single physical state. A source NaN without an
//! explanatory event maps to UnknownMissingReason; it never implies sensor
//! failure, structural absence, calibration failure, or process drift.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const OBSERVATION_FRAME_SCHEMA_V1: &str =
    "luminous.semi-bench.sensor-observation-frame.v1";
pub const OBSERVATION_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.sensor-observation-report.v1";
pub const CORRESPONDENCE_SCHEMA_V1: &str =
    "luminous.semi-bench.sensor-correspondence-manifest.v1";
pub const CORRESPONDENCE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.sensor-correspondence-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementUncertaintyV1 {
    pub standard_uncertainty: f64,
    pub coverage_probability: Option<f64>,
    pub method_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "details", rename_all = "snake_case")]
pub enum SensorObservationStateV1 {
    Observed {
        value: f64,
        unit: String,
        uncertainty: MeasurementUncertaintyV1,
    },
    /// A source value exists but units and uncertainty semantics are unknown.
    ObservedUnqualified { value: f64 },
    NotInstalled { evidence_ref: String },
    NotScheduled { evidence_ref: String },
    CommunicationLoss { evidence_ref: String },
    InvalidReading { evidence_ref: String },
    Censored {
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        unit: String,
        censoring_basis: String,
        evidence_ref: String,
    },
    UnknownMissingReason,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorObservationV1 {
    pub sensor_id: String,
    pub state: SensorObservationStateV1,
    #[serde(default)]
    pub instrument_id: Option<String>,
    #[serde(default)]
    pub calibration_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorObservationFrameV1 {
    pub schema: String,
    pub frame_id: String,
    pub subject_id: String,
    pub source_clock_domain_id: String,
    pub observed_at_source_local: Option<String>,
    pub sensor_schema_sha256: String,
    pub source_artifact_sha256: String,
    pub transform_sha256: String,
    pub observations: Vec<SensorObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationStateCountsV1 {
    pub observed: usize,
    pub observed_unqualified: usize,
    pub not_installed: usize,
    pub not_scheduled: usize,
    pub communication_loss: usize,
    pub invalid_reading: usize,
    pub censored: usize,
    pub unknown_missing_reason: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationClaimLimitsV1 {
    pub validates_representation_and_integrity_only: bool,
    pub unknown_missing_reason_does_not_infer_physical_cause: bool,
    pub observed_unqualified_is_not_a_calibrated_measurement: bool,
    pub does_not_establish_sensor_accuracy_or_calibration_truth: bool,
    pub does_not_establish_process_causality_or_fab_transfer: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorObservationReportV1 {
    pub schema: String,
    pub frame_id: String,
    pub frame_sha256: String,
    pub sensor_schema_sha256: String,
    pub source_artifact_sha256: String,
    pub transform_sha256: String,
    pub sensor_count: usize,
    pub state_counts: ObservationStateCountsV1,
    pub claim_limits: ObservationClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SensorObservationError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorCorrespondenceStatusV1 {
    Matched,
    StructurallyAbsent,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorUnitTransformV1 {
    pub source_unit: String,
    pub target_unit: String,
    /// target_value = source_value * scale + offset
    pub scale: f64,
    pub offset: f64,
    pub transform_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorCorrespondenceV1 {
    pub target_sensor_id: String,
    pub status: SensorCorrespondenceStatusV1,
    pub source_sensor_id: Option<String>,
    pub unit_transform: Option<SensorUnitTransformV1>,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorCorrespondenceManifestV1 {
    pub schema: String,
    pub mapping_id: String,
    pub source_sensor_schema_sha256: String,
    pub target_sensor_schema_sha256: String,
    pub mapping_authority_ref: String,
    pub correspondences: Vec<SensorCorrespondenceV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorCorrespondenceReportV1 {
    pub schema: String,
    pub mapping_id: String,
    pub source_sensor_schema_sha256: String,
    pub target_sensor_schema_sha256: String,
    pub matched_count: usize,
    pub structurally_absent_count: usize,
    pub unsupported_count: usize,
    pub position_only_alignment_rejected: bool,
    pub report_sha256: String,
}

fn require_text(value: &str, field: &str) -> Result<(), SensorObservationError> {
    if value.trim().is_empty() {
        return Err(SensorObservationError(format!(
            "{field}: expected non-empty string"
        )));
    }
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<(), SensorObservationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SensorObservationError(format!(
            "{field}: expected lowercase SHA-256 hex"
        )));
    }
    Ok(())
}

fn finite(value: f64, field: &str) -> Result<(), SensorObservationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SensorObservationError(format!(
            "{field}: expected finite number"
        )))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_observation(
    observation: &SensorObservationV1,
    index: usize,
) -> Result<(), SensorObservationError> {
    let prefix = format!("observations[{index}]");
    require_text(&observation.sensor_id, &format!("{prefix}.sensor_id"))?;
    if let Some(instrument_id) = observation.instrument_id.as_deref() {
        require_text(instrument_id, &format!("{prefix}.instrument_id"))?;
    }
    if let Some(calibration_ref) = observation.calibration_ref.as_deref() {
        require_text(calibration_ref, &format!("{prefix}.calibration_ref"))?;
    }

    match &observation.state {
        SensorObservationStateV1::Observed {
            value,
            unit,
            uncertainty,
        } => {
            finite(*value, &format!("{prefix}.value"))?;
            require_text(unit, &format!("{prefix}.unit"))?;
            finite(
                uncertainty.standard_uncertainty,
                &format!("{prefix}.uncertainty.standard_uncertainty"),
            )?;
            if uncertainty.standard_uncertainty < 0.0 {
                return Err(SensorObservationError(format!(
                    "{prefix}.uncertainty.standard_uncertainty: must be non-negative"
                )));
            }
            if let Some(coverage) = uncertainty.coverage_probability {
                finite(
                    coverage,
                    &format!("{prefix}.uncertainty.coverage_probability"),
                )?;
                if !(0.0..=1.0).contains(&coverage) {
                    return Err(SensorObservationError(format!(
                        "{prefix}.uncertainty.coverage_probability: must be in [0, 1]"
                    )));
                }
            }
            require_text(
                &uncertainty.method_ref,
                &format!("{prefix}.uncertainty.method_ref"),
            )?;
        }
        SensorObservationStateV1::ObservedUnqualified { value } => {
            finite(*value, &format!("{prefix}.value"))?;
        }
        SensorObservationStateV1::NotInstalled { evidence_ref }
        | SensorObservationStateV1::NotScheduled { evidence_ref }
        | SensorObservationStateV1::CommunicationLoss { evidence_ref }
        | SensorObservationStateV1::InvalidReading { evidence_ref } => {
            require_text(evidence_ref, &format!("{prefix}.evidence_ref"))?;
        }
        SensorObservationStateV1::Censored {
            lower_bound,
            upper_bound,
            unit,
            censoring_basis,
            evidence_ref,
        } => {
            if lower_bound.is_none() && upper_bound.is_none() {
                return Err(SensorObservationError(format!(
                    "{prefix}: censored observation requires at least one bound"
                )));
            }
            if let Some(value) = lower_bound {
                finite(*value, &format!("{prefix}.lower_bound"))?;
            }
            if let Some(value) = upper_bound {
                finite(*value, &format!("{prefix}.upper_bound"))?;
            }
            if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
                if lower > upper {
                    return Err(SensorObservationError(format!(
                        "{prefix}: lower_bound must not exceed upper_bound"
                    )));
                }
            }
            require_text(unit, &format!("{prefix}.unit"))?;
            require_text(censoring_basis, &format!("{prefix}.censoring_basis"))?;
            require_text(evidence_ref, &format!("{prefix}.evidence_ref"))?;
        }
        SensorObservationStateV1::UnknownMissingReason => {}
    }
    Ok(())
}

/// Validates a typed frame without guessing physical semantics.
pub fn validate_sensor_observation_frame_v1(
    frame: &SensorObservationFrameV1,
) -> Result<SensorObservationReportV1, SensorObservationError> {
    if frame.schema != OBSERVATION_FRAME_SCHEMA_V1 {
        return Err(SensorObservationError(
            "schema: unsupported or missing observation frame".into(),
        ));
    }
    require_text(&frame.frame_id, "frame_id")?;
    require_text(&frame.subject_id, "subject_id")?;
    require_text(&frame.source_clock_domain_id, "source_clock_domain_id")?;
    if let Some(timestamp) = frame.observed_at_source_local.as_deref() {
        require_text(timestamp, "observed_at_source_local")?;
    }
    require_sha256(&frame.sensor_schema_sha256, "sensor_schema_sha256")?;
    require_sha256(&frame.source_artifact_sha256, "source_artifact_sha256")?;
    require_sha256(&frame.transform_sha256, "transform_sha256")?;
    if frame.observations.is_empty() {
        return Err(SensorObservationError(
            "observations: expected non-empty array".into(),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut counts = ObservationStateCountsV1 {
        observed: 0,
        observed_unqualified: 0,
        not_installed: 0,
        not_scheduled: 0,
        communication_loss: 0,
        invalid_reading: 0,
        censored: 0,
        unknown_missing_reason: 0,
    };
    for (index, observation) in frame.observations.iter().enumerate() {
        validate_observation(observation, index)?;
        if !seen.insert(observation.sensor_id.as_str()) {
            return Err(SensorObservationError(format!(
                "observations[{index}].sensor_id: duplicate sensor identity"
            )));
        }
        match &observation.state {
            SensorObservationStateV1::Observed { .. } => counts.observed += 1,
            SensorObservationStateV1::ObservedUnqualified { .. } => {
                counts.observed_unqualified += 1;
            }
            SensorObservationStateV1::NotInstalled { .. } => counts.not_installed += 1,
            SensorObservationStateV1::NotScheduled { .. } => counts.not_scheduled += 1,
            SensorObservationStateV1::CommunicationLoss { .. } => {
                counts.communication_loss += 1;
            }
            SensorObservationStateV1::InvalidReading { .. } => counts.invalid_reading += 1,
            SensorObservationStateV1::Censored { .. } => counts.censored += 1,
            SensorObservationStateV1::UnknownMissingReason => {
                counts.unknown_missing_reason += 1;
            }
        }
    }

    let serialized = serde_json::to_vec(frame)
        .map_err(|error| SensorObservationError(format!("frame serialization failed: {error}")))?;
    Ok(SensorObservationReportV1 {
        schema: OBSERVATION_REPORT_SCHEMA_V1.into(),
        frame_id: frame.frame_id.clone(),
        frame_sha256: sha256_hex(&serialized),
        sensor_schema_sha256: frame.sensor_schema_sha256.clone(),
        source_artifact_sha256: frame.source_artifact_sha256.clone(),
        transform_sha256: frame.transform_sha256.clone(),
        sensor_count: frame.observations.len(),
        state_counts: counts,
        claim_limits: ObservationClaimLimitsV1 {
            validates_representation_and_integrity_only: true,
            unknown_missing_reason_does_not_infer_physical_cause: true,
            observed_unqualified_is_not_a_calibrated_measurement: true,
            does_not_establish_sensor_accuracy_or_calibration_truth: true,
            does_not_establish_process_causality_or_fab_transfer: true,
        },
    })
}

/// Converts one SECOM feature row into explicitly unqualified observations.
///
/// SECOM columns are anonymous and have no unit, calibration provenance or
/// missing-reason metadata. Numeric values remain unqualified; missing cells
/// map only to UnknownMissingReason.
pub fn adapt_secom_row_to_observation_frame_v1(
    dataset_id: &str,
    sample_id: &str,
    source_clock_domain_id: &str,
    observed_at_source_local: &str,
    feature_values: &[Option<f64>],
    source_artifact_sha256: &str,
    transform_sha256: &str,
) -> Result<SensorObservationFrameV1, SensorObservationError> {
    require_text(dataset_id, "dataset_id")?;
    require_text(sample_id, "sample_id")?;
    require_text(source_clock_domain_id, "source_clock_domain_id")?;
    require_text(observed_at_source_local, "observed_at_source_local")?;
    require_sha256(source_artifact_sha256, "source_artifact_sha256")?;
    require_sha256(transform_sha256, "transform_sha256")?;
    if feature_values.is_empty() {
        return Err(SensorObservationError(
            "feature_values: expected non-empty array".into(),
        ));
    }

    let sensor_ids: Vec<String> = (0..feature_values.len())
        .map(|index| format!("secom-anonymous-column-{index:04}"))
        .collect();
    let sensor_schema_sha256 = sha256_hex(
        &serde_json::to_vec(&sensor_ids).map_err(|error| {
            SensorObservationError(format!("sensor schema serialization failed: {error}"))
        })?,
    );
    let observations = sensor_ids
        .into_iter()
        .zip(feature_values)
        .map(|(sensor_id, value)| SensorObservationV1 {
            sensor_id,
            state: match value {
                Some(value) => SensorObservationStateV1::ObservedUnqualified { value: *value },
                None => SensorObservationStateV1::UnknownMissingReason,
            },
            instrument_id: None,
            calibration_ref: None,
        })
        .collect();

    Ok(SensorObservationFrameV1 {
        schema: OBSERVATION_FRAME_SCHEMA_V1.into(),
        frame_id: format!("{dataset_id}:{sample_id}"),
        subject_id: sample_id.into(),
        source_clock_domain_id: source_clock_domain_id.into(),
        observed_at_source_local: Some(observed_at_source_local.into()),
        sensor_schema_sha256,
        source_artifact_sha256: source_artifact_sha256.into(),
        transform_sha256: transform_sha256.into(),
        observations,
    })
}

pub fn validate_sensor_correspondence_manifest_v1(
    manifest: &SensorCorrespondenceManifestV1,
) -> Result<SensorCorrespondenceReportV1, SensorObservationError> {
    if manifest.schema != CORRESPONDENCE_SCHEMA_V1 {
        return Err(SensorObservationError(
            "schema: unsupported or missing correspondence manifest".into(),
        ));
    }
    require_text(&manifest.mapping_id, "mapping_id")?;
    require_sha256(
        &manifest.source_sensor_schema_sha256,
        "source_sensor_schema_sha256",
    )?;
    require_sha256(
        &manifest.target_sensor_schema_sha256,
        "target_sensor_schema_sha256",
    )?;
    require_text(&manifest.mapping_authority_ref, "mapping_authority_ref")?;
    if manifest.correspondences.is_empty() {
        return Err(SensorObservationError(
            "correspondences: expected non-empty array".into(),
        ));
    }

    let mut target_ids = BTreeSet::new();
    let mut source_ids = BTreeSet::new();
    let mut matched_count = 0;
    let mut structurally_absent_count = 0;
    let mut unsupported_count = 0;
    for (index, correspondence) in manifest.correspondences.iter().enumerate() {
        require_text(
            &correspondence.target_sensor_id,
            &format!("correspondences[{index}].target_sensor_id"),
        )?;
        require_text(
            &correspondence.evidence_ref,
            &format!("correspondences[{index}].evidence_ref"),
        )?;
        if !target_ids.insert(correspondence.target_sensor_id.as_str()) {
            return Err(SensorObservationError(format!(
                "correspondences[{index}].target_sensor_id: duplicate target sensor"
            )));
        }

        match &correspondence.status {
            SensorCorrespondenceStatusV1::Matched => {
                let source_id = correspondence.source_sensor_id.as_deref().ok_or_else(|| {
                    SensorObservationError(format!(
                        "correspondences[{index}].source_sensor_id: required for matched status"
                    ))
                })?;
                require_text(source_id, &format!("correspondences[{index}].source_sensor_id"))?;
                if !source_ids.insert(source_id) {
                    return Err(SensorObservationError(format!(
                        "correspondences[{index}].source_sensor_id: duplicate source sensor mapping"
                    )));
                }
                let transform = correspondence.unit_transform.as_ref().ok_or_else(|| {
                    SensorObservationError(format!(
                        "correspondences[{index}].unit_transform: required for matched status"
                    ))
                })?;
                require_text(
                    &transform.source_unit,
                    &format!("correspondences[{index}].unit_transform.source_unit"),
                )?;
                require_text(
                    &transform.target_unit,
                    &format!("correspondences[{index}].unit_transform.target_unit"),
                )?;
                finite(
                    transform.scale,
                    &format!("correspondences[{index}].unit_transform.scale"),
                )?;
                if transform.scale == 0.0 {
                    return Err(SensorObservationError(format!(
                        "correspondences[{index}].unit_transform.scale: must be non-zero"
                    )));
                }
                finite(
                    transform.offset,
                    &format!("correspondences[{index}].unit_transform.offset"),
                )?;
                require_text(
                    &transform.transform_evidence_ref,
                    &format!("correspondences[{index}].unit_transform.transform_evidence_ref"),
                )?;
                matched_count += 1;
            }
            SensorCorrespondenceStatusV1::StructurallyAbsent => {
                if correspondence.source_sensor_id.is_some()
                    || correspondence.unit_transform.is_some()
                {
                    return Err(SensorObservationError(format!(
                        "correspondences[{index}]: structurally_absent must not have source identity or unit transform"
                    )));
                }
                structurally_absent_count += 1;
            }
            SensorCorrespondenceStatusV1::Unsupported => {
                if correspondence.source_sensor_id.is_some()
                    || correspondence.unit_transform.is_some()
                {
                    return Err(SensorObservationError(format!(
                        "correspondences[{index}]: unsupported mapping must not imply source identity or unit transform"
                    )));
                }
                unsupported_count += 1;
            }
        }
    }

    let serialized = serde_json::to_vec(manifest).map_err(|error| {
        SensorObservationError(format!("correspondence serialization failed: {error}"))
    })?;
    Ok(SensorCorrespondenceReportV1 {
        schema: CORRESPONDENCE_REPORT_SCHEMA_V1.into(),
        mapping_id: manifest.mapping_id.clone(),
        source_sensor_schema_sha256: manifest.source_sensor_schema_sha256.clone(),
        target_sensor_schema_sha256: manifest.target_sensor_schema_sha256.clone(),
        matched_count,
        structurally_absent_count,
        unsupported_count,
        position_only_alignment_rejected: true,
        report_sha256: sha256_hex(&serialized),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn frame(state: SensorObservationStateV1) -> SensorObservationFrameV1 {
        SensorObservationFrameV1 {
            schema: OBSERVATION_FRAME_SCHEMA_V1.into(),
            frame_id: "frame-1".into(),
            subject_id: "wafer-1".into(),
            source_clock_domain_id: "fab-a-local".into(),
            observed_at_source_local: Some("2026-01-01 12:00:00".into()),
            sensor_schema_sha256: SHA_A.into(),
            source_artifact_sha256: SHA_B.into(),
            transform_sha256: SHA_C.into(),
            observations: vec![SensorObservationV1 {
                sensor_id: "sensor-pressure-1".into(),
                state,
                instrument_id: Some("instrument-1".into()),
                calibration_ref: Some("calibration-1".into()),
            }],
        }
    }

    fn correspondence_manifest() -> SensorCorrespondenceManifestV1 {
        SensorCorrespondenceManifestV1 {
            schema: CORRESPONDENCE_SCHEMA_V1.into(),
            mapping_id: "map-v1".into(),
            source_sensor_schema_sha256: SHA_A.into(),
            target_sensor_schema_sha256: SHA_B.into(),
            mapping_authority_ref: "approved-map-17".into(),
            correspondences: vec![
                SensorCorrespondenceV1 {
                    target_sensor_id: "target-pressure".into(),
                    status: SensorCorrespondenceStatusV1::Matched,
                    source_sensor_id: Some("source-pressure".into()),
                    unit_transform: Some(SensorUnitTransformV1 {
                        source_unit: "kPa".into(),
                        target_unit: "Pa".into(),
                        scale: 1000.0,
                        offset: 0.0,
                        transform_evidence_ref: "unit-proof-1".into(),
                    }),
                    evidence_ref: "semantic-map-evidence-1".into(),
                },
                SensorCorrespondenceV1 {
                    target_sensor_id: "target-flow".into(),
                    status: SensorCorrespondenceStatusV1::StructurallyAbsent,
                    source_sensor_id: None,
                    unit_transform: None,
                    evidence_ref: "tool-config-evidence-2".into(),
                },
                SensorCorrespondenceV1 {
                    target_sensor_id: "target-unknown".into(),
                    status: SensorCorrespondenceStatusV1::Unsupported,
                    source_sensor_id: None,
                    unit_transform: None,
                    evidence_ref: "mapping-gap-3".into(),
                },
            ],
        }
    }

    #[test]
    fn qualified_observation_requires_unit_and_uncertainty() {
        let report = validate_sensor_observation_frame_v1(&frame(
            SensorObservationStateV1::Observed {
                value: 1.5,
                unit: "kPa".into(),
                uncertainty: MeasurementUncertaintyV1 {
                    standard_uncertainty: 0.02,
                    coverage_probability: Some(0.95),
                    method_ref: "instrument-uncertainty-report-3".into(),
                },
            },
        ))
        .expect("qualified observation validates");
        assert_eq!(report.state_counts.observed, 1);
        assert!(report.claim_limits.observed_unqualified_is_not_a_calibrated_measurement);
    }

    #[test]
    fn source_missing_value_maps_only_to_unknown_missing_reason() {
        let values = [Some(3.2), None, Some(0.0)];
        let adapted = adapt_secom_row_to_observation_frame_v1(
            "SECOM-mirror",
            "sample-17",
            "secom-naive-source-clock",
            "19/07/2008 12:15:00",
            &values,
            SHA_B,
            SHA_C,
        )
        .expect("SECOM adapter succeeds");
        let report = validate_sensor_observation_frame_v1(&adapted)
            .expect("adapted frame validates");
        assert_eq!(report.state_counts.observed_unqualified, 2);
        assert_eq!(report.state_counts.unknown_missing_reason, 1);
        assert_eq!(report.state_counts.communication_loss, 0);
        assert_eq!(report.state_counts.not_installed, 0);
        assert_eq!(report.state_counts.invalid_reading, 0);
        assert!(adapted.observations[1].instrument_id.is_none());
        assert!(adapted.observations[1].calibration_ref.is_none());
    }

    #[test]
    fn non_finite_qualified_value_is_rejected() {
        let input = frame(SensorObservationStateV1::Observed {
            value: f64::NAN,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.1,
                coverage_probability: Some(0.95),
                method_ref: "method-1".into(),
            },
        });
        assert!(validate_sensor_observation_frame_v1(&input).is_err());
    }

    #[test]
    fn negative_uncertainty_is_rejected() {
        let input = frame(SensorObservationStateV1::Observed {
            value: 1.0,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: -0.1,
                coverage_probability: None,
                method_ref: "method-1".into(),
            },
        });
        assert!(validate_sensor_observation_frame_v1(&input).is_err());
    }

    #[test]
    fn censored_observation_requires_valid_bounds_and_evidence() {
        let valid = frame(SensorObservationStateV1::Censored {
            lower_bound: Some(0.0),
            upper_bound: None,
            unit: "nm".into(),
            censoring_basis: "below detection threshold".into(),
            evidence_ref: "instrument-limit-record-1".into(),
        });
        assert!(validate_sensor_observation_frame_v1(&valid).is_ok());

        let invalid = frame(SensorObservationStateV1::Censored {
            lower_bound: Some(3.0),
            upper_bound: Some(2.0),
            unit: "nm".into(),
            censoring_basis: "detection range".into(),
            evidence_ref: "instrument-limit-record-1".into(),
        });
        assert!(validate_sensor_observation_frame_v1(&invalid).is_err());
    }

    #[test]
    fn reasoned_missing_states_require_evidence_reference() {
        let invalid = frame(SensorObservationStateV1::CommunicationLoss {
            evidence_ref: " ".into(),
        });
        assert!(validate_sensor_observation_frame_v1(&invalid).is_err());

        let valid = frame(SensorObservationStateV1::NotScheduled {
            evidence_ref: "sampling-plan-17".into(),
        });
        assert!(validate_sensor_observation_frame_v1(&valid).is_ok());
    }

    #[test]
    fn duplicate_sensor_identity_is_rejected() {
        let mut input = frame(SensorObservationStateV1::UnknownMissingReason);
        input.observations.push(input.observations[0].clone());
        assert!(validate_sensor_observation_frame_v1(&input).is_err());
    }

    #[test]
    fn matched_correspondence_requires_identity_and_explicit_unit_transform() {
        let report = validate_sensor_correspondence_manifest_v1(&correspondence_manifest())
            .expect("correspondence manifest validates");
        assert_eq!(report.matched_count, 1);
        assert_eq!(report.structurally_absent_count, 1);
        assert_eq!(report.unsupported_count, 1);
        assert!(report.position_only_alignment_rejected);
    }

    #[test]
    fn structural_absence_cannot_be_confused_with_measurement_loss() {
        let mut input = correspondence_manifest();
        input.correspondences[1].source_sensor_id = Some("sensor-2".into());
        assert!(validate_sensor_correspondence_manifest_v1(&input).is_err());
    }

    #[test]
    fn position_only_mapping_is_rejected() {
        let mut input = correspondence_manifest();
        input.correspondences[0].source_sensor_id = None;
        assert!(validate_sensor_correspondence_manifest_v1(&input).is_err());
    }

    #[test]
    fn zero_unit_scale_is_rejected() {
        let mut input = correspondence_manifest();
        input.correspondences[0]
            .unit_transform
            .as_mut()
            .unwrap()
            .scale = 0.0;
        assert!(validate_sensor_correspondence_manifest_v1(&input).is_err());
    }
}
