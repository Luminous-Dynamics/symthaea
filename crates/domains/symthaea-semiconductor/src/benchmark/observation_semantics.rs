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

pub const OBSERVATION_FRAME_SCHEMA_V1: &str = "luminous.semi-bench.sensor-observation-frame.v1";
pub const OBSERVATION_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.sensor-observation-report.v1";
pub const CORRESPONDENCE_SCHEMA_V1: &str = "luminous.semi-bench.sensor-correspondence-manifest.v1";
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
    ObservedUnqualified {
        value: f64,
    },
    NotInstalled {
        evidence_ref: String,
    },
    NotScheduled {
        evidence_ref: String,
    },
    CommunicationLoss {
        evidence_ref: String,
    },
    InvalidReading {
        evidence_ref: String,
    },
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorCorrespondenceExecutionReportV1 {
    pub schema: String,
    pub mapping_id: String,
    pub source_frame_sha256: String,
    pub correspondence_manifest_sha256: String,
    pub target_frame_sha256: Option<String>,
    pub matched_count: usize,
    pub structurally_absent_count: usize,
    pub unsupported_count: usize,
    pub transformed_observed_count: usize,
    pub copied_non_point_state_count: usize,
    pub unmapped_source_sensor_ids: Vec<String>,
    pub abstain_required: bool,
    pub abstention_reasons: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorCorrespondenceExecutionV1 {
    /// None means a complete target frame could not be constructed safely.
    pub target_frame: Option<SensorObservationFrameV1>,
    pub report: SensorCorrespondenceExecutionReportV1,
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
            if let (Some(lower), Some(upper)) = (lower_bound, upper_bound)
                && lower > upper
            {
                return Err(SensorObservationError(format!(
                    "{prefix}: lower_bound must not exceed upper_bound"
                )));
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
    let sensor_schema_sha256 = sha256_hex(&serde_json::to_vec(&sensor_ids).map_err(|error| {
        SensorObservationError(format!("sensor schema serialization failed: {error}"))
    })?);
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
                require_text(
                    source_id,
                    &format!("correspondences[{index}].source_sensor_id"),
                )?;
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

fn transform_observation_state_v1(
    source: &SensorObservationV1,
    mapping: &SensorCorrespondenceV1,
    index: usize,
) -> Result<(SensorObservationStateV1, bool), SensorObservationError> {
    let transform = mapping.unit_transform.as_ref().ok_or_else(|| {
        SensorObservationError(format!(
            "correspondences[{index}].unit_transform: missing for matched mapping"
        ))
    })?;
    let abstain = |reason: String| {
        Err(SensorObservationError(format!(
            "abstain: {} -> {}: {reason}",
            source.sensor_id, mapping.target_sensor_id
        )))
    };

    let state = match &source.state {
        SensorObservationStateV1::Observed {
            value,
            unit,
            uncertainty,
        } => {
            if unit != &transform.source_unit {
                return abstain(format!(
                    "source unit {unit} differs from declared transform source unit {}",
                    transform.source_unit
                ));
            }
            let converted_value = value * transform.scale + transform.offset;
            let converted_uncertainty =
                uncertainty.standard_uncertainty * transform.scale.abs();
            if !converted_value.is_finite() || !converted_uncertainty.is_finite() {
                return abstain("unit conversion overflowed to a non-finite value".into());
            }
            (
                SensorObservationStateV1::Observed {
                    value: converted_value,
                    unit: transform.target_unit.clone(),
                    uncertainty: MeasurementUncertaintyV1 {
                        standard_uncertainty: converted_uncertainty,
                        coverage_probability: uncertainty.coverage_probability,
                        method_ref: format!(
                            "{};transform={}",
                            uncertainty.method_ref, transform.transform_evidence_ref
                        ),
                    },
                },
                true,
            )
        }
        SensorObservationStateV1::ObservedUnqualified { .. } => {
            return abstain(
                "unqualified observation has no source unit or uncertainty semantics".into(),
            );
        }
        SensorObservationStateV1::Censored {
            lower_bound,
            upper_bound,
            unit,
            censoring_basis,
            evidence_ref,
        } => {
            if unit != &transform.source_unit {
                return abstain(format!(
                    "censored source unit {unit} differs from declared transform source unit {}",
                    transform.source_unit
                ));
            }
            let convert = |value: f64| value * transform.scale + transform.offset;
            let converted_bounds = match (lower_bound, upper_bound) {
                (Some(lower), Some(upper)) if transform.scale > 0.0 => {
                    (Some(convert(*lower)), Some(convert(*upper)))
                }
                (Some(lower), Some(upper)) => (Some(convert(*upper)), Some(convert(*lower))),
                (Some(lower), None) if transform.scale > 0.0 => (Some(convert(*lower)), None),
                (Some(lower), None) => (None, Some(convert(*lower))),
                (None, Some(upper)) if transform.scale > 0.0 => (None, Some(convert(*upper))),
                (None, Some(upper)) => (Some(convert(*upper)), None),
                (None, None) => {
                    return abstain("censored observation has no numeric bounds".into());
                }
            };
            if converted_bounds
                .0
                .into_iter()
                .chain(converted_bounds.1)
                .any(|value| !value.is_finite())
            {
                return abstain("censored unit conversion produced a non-finite bound".into());
            }
            (
                SensorObservationStateV1::Censored {
                    lower_bound: converted_bounds.0,
                    upper_bound: converted_bounds.1,
                    unit: transform.target_unit.clone(),
                    censoring_basis: censoring_basis.clone(),
                    evidence_ref: format!(
                        "{};transform={}",
                        evidence_ref, transform.transform_evidence_ref
                    ),
                },
                true,
            )
        }
        other => ((*other).clone(), false),
    };
    Ok(state)
}

/// Executes a correspondence manifest against one source frame.
///
/// No positional fallback is permitted. A missing source identity, unsupported
/// target mapping, unqualified numeric source value, or incompatible unit
/// causes an explicit abstention and returns no target frame.
pub fn execute_sensor_correspondence_v1(
    source_frame: &SensorObservationFrameV1,
    manifest: &SensorCorrespondenceManifestV1,
) -> Result<SensorCorrespondenceExecutionV1, SensorObservationError> {
    validate_sensor_observation_frame_v1(source_frame)?;
    validate_sensor_correspondence_manifest_v1(manifest)?;
    if source_frame.sensor_schema_sha256 != manifest.source_sensor_schema_sha256 {
        return Err(SensorObservationError(
            "source frame sensor schema digest differs from correspondence manifest".into(),
        ));
    }

    let source_frame_bytes = serde_json::to_vec(source_frame).map_err(|error| {
        SensorObservationError(format!("source frame serialization failed: {error}"))
    })?;
    let mut canonical_manifest = manifest.clone();
    canonical_manifest
        .correspondences
        .sort_by(|left, right| left.target_sensor_id.cmp(&right.target_sensor_id));
    let manifest_bytes = serde_json::to_vec(&canonical_manifest).map_err(|error| {
        SensorObservationError(format!("correspondence manifest serialization failed: {error}"))
    })?;
    let source_frame_sha256 = sha256_hex(&source_frame_bytes);
    let correspondence_manifest_sha256 = sha256_hex(&manifest_bytes);
    let source_by_id: std::collections::BTreeMap<&str, &SensorObservationV1> = source_frame
        .observations
        .iter()
        .map(|observation| (observation.sensor_id.as_str(), observation))
        .collect();

    let mut ordered_correspondences: Vec<_> = manifest.correspondences.iter().collect();
    ordered_correspondences.sort_by(|left, right| {
        left.target_sensor_id.cmp(&right.target_sensor_id)
    });

    let mapped_source_ids: BTreeSet<&str> = manifest
        .correspondences
        .iter()
        .filter_map(|mapping| mapping.source_sensor_id.as_deref())
        .collect();
    let mut unmapped_source_sensor_ids: Vec<String> = source_frame
        .observations
        .iter()
        .filter(|observation| !mapped_source_ids.contains(observation.sensor_id.as_str()))
        .map(|observation| observation.sensor_id.clone())
        .collect();
    unmapped_source_sensor_ids.sort();

    let mut target_observations = Vec::with_capacity(ordered_correspondences.len());
    let mut matched_count = 0_usize;
    let mut structurally_absent_count = 0_usize;
    let mut unsupported_count = 0_usize;
    let mut transformed_observed_count = 0_usize;
    let mut copied_non_point_state_count = 0_usize;
    let mut abstention_reasons = Vec::new();

    for mapping in ordered_correspondences {
        match &mapping.status {
            SensorCorrespondenceStatusV1::Matched => {
                matched_count += 1;
                let Some(source_id) = mapping.source_sensor_id.as_deref() else {
                    abstention_reasons.push(format!(
                        "target {} declares matched status without a source sensor ID",
                        mapping.target_sensor_id
                    ));
                    continue;
                };
                let Some(source) = source_by_id.get(source_id).copied() else {
                    abstention_reasons.push(format!(
                        "target {} requires source sensor {}, which is absent from the source frame",
                        mapping.target_sensor_id, source_id
                    ));
                    continue;
                };
                match transform_observation_state_v1(source, mapping, 0) {
                    Ok((state, transformed)) => {
                        if transformed {
                            transformed_observed_count += 1;
                        } else {
                            copied_non_point_state_count += 1;
                        }
                        target_observations.push(SensorObservationV1 {
                            sensor_id: mapping.target_sensor_id.clone(),
                            state,
                            instrument_id: source.instrument_id.clone(),
                            calibration_ref: source.calibration_ref.clone(),
                        });
                    }
                    Err(error) => abstention_reasons.push(error.to_string()),
                }
            }
            SensorCorrespondenceStatusV1::StructurallyAbsent => {
                structurally_absent_count += 1;
                target_observations.push(SensorObservationV1 {
                    sensor_id: mapping.target_sensor_id.clone(),
                    state: SensorObservationStateV1::NotInstalled {
                        evidence_ref: mapping.evidence_ref.clone(),
                    },
                    instrument_id: None,
                    calibration_ref: None,
                });
            }
            SensorCorrespondenceStatusV1::Unsupported => {
                unsupported_count += 1;
                abstention_reasons.push(format!(
                    "target {} has unsupported correspondence: {}",
                    mapping.target_sensor_id, mapping.evidence_ref
                ));
            }
        }
    }

    let abstain_required = !abstention_reasons.is_empty();
    let target_frame = if abstain_required {
        None
    } else {
        let transform_material = (
            source_frame.transform_sha256.as_str(),
            correspondence_manifest_sha256.as_str(),
        );
        let transform_sha256 = sha256_hex(
            &serde_json::to_vec(&transform_material).map_err(|error| {
                SensorObservationError(format!("transform digest serialization failed: {error}"))
            })?,
        );
        let frame = SensorObservationFrameV1 {
            schema: OBSERVATION_FRAME_SCHEMA_V1.into(),
            frame_id: format!("{}::{}", source_frame.frame_id, manifest.mapping_id),
            subject_id: source_frame.subject_id.clone(),
            source_clock_domain_id: source_frame.source_clock_domain_id.clone(),
            observed_at_source_local: source_frame.observed_at_source_local.clone(),
            sensor_schema_sha256: manifest.target_sensor_schema_sha256.clone(),
            source_artifact_sha256: source_frame.source_artifact_sha256.clone(),
            transform_sha256,
            observations: target_observations,
        };
        validate_sensor_observation_frame_v1(&frame)?;
        Some(frame)
    };

    let target_frame_sha256 = target_frame
        .as_ref()
        .map(|frame| {
            serde_json::to_vec(frame)
                .map(|bytes| sha256_hex(&bytes))
                .map_err(|error| {
                    SensorObservationError(format!("target frame serialization failed: {error}"))
                })
        })
        .transpose()?;
    let mut report = SensorCorrespondenceExecutionReportV1 {
        schema: "luminous.semi-bench.sensor-correspondence-execution-report.v1".into(),
        mapping_id: manifest.mapping_id.clone(),
        source_frame_sha256,
        correspondence_manifest_sha256,
        target_frame_sha256,
        matched_count,
        structurally_absent_count,
        unsupported_count,
        transformed_observed_count,
        copied_non_point_state_count,
        unmapped_source_sensor_ids,
        abstain_required,
        abstention_reasons,
        report_sha256: String::new(),
    };
    let mut material = report.clone();
    material.report_sha256.clear();
    report.report_sha256 = sha256_hex(
        &serde_json::to_vec(&material).map_err(|error| {
            SensorObservationError(format!("execution report serialization failed: {error}"))
        })?,
    );
    Ok(SensorCorrespondenceExecutionV1 {
        target_frame,
        report,
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
        let report =
            validate_sensor_observation_frame_v1(&frame(SensorObservationStateV1::Observed {
                value: 1.5,
                unit: "kPa".into(),
                uncertainty: MeasurementUncertaintyV1 {
                    standard_uncertainty: 0.02,
                    coverage_probability: Some(0.95),
                    method_ref: "instrument-uncertainty-report-3".into(),
                },
            }))
            .expect("qualified observation validates");
        assert_eq!(report.state_counts.observed, 1);
        assert!(
            report
                .claim_limits
                .observed_unqualified_is_not_a_calibrated_measurement
        );
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
        let report =
            validate_sensor_observation_frame_v1(&adapted).expect("adapted frame validates");
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

    fn mapped_source_frame(state: SensorObservationStateV1) -> SensorObservationFrameV1 {
        let mut input = frame(state);
        input.observations[0].sensor_id = "source-pressure".into();
        input
    }

    fn complete_correspondence_manifest() -> SensorCorrespondenceManifestV1 {
        let mut manifest = correspondence_manifest();
        manifest
            .correspondences
            .retain(|mapping| mapping.target_sensor_id != "target-unknown");
        manifest
    }

    #[test]
    fn correspondence_execution_converts_value_and_propagates_uncertainty() {
        let source = mapped_source_frame(SensorObservationStateV1::Observed {
            value: 2.5,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.02,
                coverage_probability: Some(0.95),
                method_ref: "gauge-report-4".into(),
            },
        });
        let manifest = complete_correspondence_manifest();
        let execution = execute_sensor_correspondence_v1(&source, &manifest).unwrap();
        assert!(!execution.report.abstain_required);
        assert_eq!(execution.report.matched_count, 1);
        assert_eq!(execution.report.structurally_absent_count, 1);
        assert!(execution.target_frame.is_some());

        let target = execution.target_frame.unwrap();
        let absent = target
            .observations
            .iter()
            .find(|observation| observation.sensor_id == "target-flow")
            .unwrap();
        assert!(matches!(
            &absent.state,
            SensorObservationStateV1::NotInstalled { .. }
        ));

        let pressure = target
            .observations
            .iter()
            .find(|observation| observation.sensor_id == "target-pressure")
            .unwrap();
        match &pressure.state {
            SensorObservationStateV1::Observed {
                value,
                unit,
                uncertainty,
            } => {
                assert_eq!(*value, 2500.0);
                assert_eq!(unit, "Pa");
                assert!((uncertainty.standard_uncertainty - 20.0).abs() < 1.0e-12);
                assert_eq!(uncertainty.coverage_probability, Some(0.95));
                assert!(uncertainty.method_ref.contains("unit-proof-1"));
            }
            other => panic!("expected transformed qualified observation, got {other:?}"),
        }
        assert_eq!(pressure.instrument_id.as_deref(), Some("instrument-1"));
        assert_eq!(pressure.calibration_ref.as_deref(), Some("calibration-1"));
        assert!(target.observations.iter().all(|observation| {
            observation.sensor_id.starts_with("target-")
        }));
    }

    #[test]
    fn reasoned_missing_and_unknown_states_survive_correspondence_execution() {
        let mut source = mapped_source_frame(SensorObservationStateV1::CommunicationLoss {
            evidence_ref: "transport-gap-22".into(),
        });
        source.observations.push(SensorObservationV1 {
            sensor_id: "source-flow".into(),
            state: SensorObservationStateV1::UnknownMissingReason,
            instrument_id: None,
            calibration_ref: None,
        });
        let mut manifest = complete_correspondence_manifest();
        manifest.correspondences.push(SensorCorrespondenceV1 {
            target_sensor_id: "target-flow-meter".into(),
            status: SensorCorrespondenceStatusV1::Matched,
            source_sensor_id: Some("source-flow".into()),
            unit_transform: Some(SensorUnitTransformV1 {
                source_unit: "kPa".into(),
                target_unit: "Pa".into(),
                scale: 1000.0,
                offset: 0.0,
                transform_evidence_ref: "flow-unit-proof".into(),
            }),
            evidence_ref: "flow-correspondence-evidence".into(),
        });

        let execution = execute_sensor_correspondence_v1(&source, &manifest).unwrap();
        let target = execution.target_frame.unwrap();
        let pressure = target
            .observations
            .iter()
            .find(|observation| observation.sensor_id == "target-pressure")
            .unwrap();
        let flow = target
            .observations
            .iter()
            .find(|observation| observation.sensor_id == "target-flow-meter")
            .unwrap();
        assert!(matches!(
            &pressure.state,
            SensorObservationStateV1::CommunicationLoss {
                evidence_ref
            } if evidence_ref == "transport-gap-22"
        ));
        assert!(matches!(
            &flow.state,
            SensorObservationStateV1::UnknownMissingReason
        ));
        assert!(!execution.report.abstain_required);
        assert_eq!(execution.report.copied_non_point_state_count, 2);
    }

    #[test]
    fn unsupported_mapping_abstains_instead_of_emitting_partial_target_frame() {
        let source = mapped_source_frame(SensorObservationStateV1::Observed {
            value: 2.5,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.02,
                coverage_probability: Some(0.95),
                method_ref: "gauge-report-4".into(),
            },
        });
        let manifest = correspondence_manifest();
        let execution = execute_sensor_correspondence_v1(&source, &manifest).unwrap();
        assert!(execution.report.abstain_required);
        assert!(execution.target_frame.is_none());
        assert_eq!(execution.report.unsupported_count, 1);
        assert!(
            execution
                .report
                .abstention_reasons
                .iter()
                .any(|reason| reason.contains("target-unknown"))
        );
    }

    #[test]
    fn unqualified_numeric_observation_cannot_cross_unit_mapping() {
        let source = mapped_source_frame(SensorObservationStateV1::ObservedUnqualified {
            value: 2.5,
        });
        let execution =
            execute_sensor_correspondence_v1(&source, &complete_correspondence_manifest())
                .unwrap();
        assert!(execution.report.abstain_required);
        assert!(execution.target_frame.is_none());
        assert!(
            execution
                .report
                .abstention_reasons
                .iter()
                .any(|reason| reason.contains("no source unit or uncertainty"))
        );
    }

    #[test]
    fn censored_bounds_transform_correctly_under_negative_scale() {
        let source = mapped_source_frame(SensorObservationStateV1::Censored {
            lower_bound: Some(1.0),
            upper_bound: Some(4.0),
            unit: "raw_unit".into(),
            censoring_basis: "instrument range limit".into(),
            evidence_ref: "censoring-record-8".into(),
        });
        let mut manifest = complete_correspondence_manifest();
        let transform = manifest.correspondences[0]
            .unit_transform
            .as_mut()
            .unwrap();
        transform.source_unit = "raw_unit".into();
        transform.target_unit = "corrected_unit".into();
        transform.scale = -2.0;
        transform.offset = 10.0;

        let execution = execute_sensor_correspondence_v1(&source, &manifest).unwrap();
        let target = execution.target_frame.unwrap();
        let pressure = target
            .observations
            .iter()
            .find(|observation| observation.sensor_id == "target-pressure")
            .unwrap();
        match &pressure.state {
            SensorObservationStateV1::Censored {
                lower_bound,
                upper_bound,
                unit,
                ..
            } => {
                assert_eq!(*lower_bound, Some(2.0));
                assert_eq!(*upper_bound, Some(8.0));
                assert_eq!(unit, "corrected_unit");
            }
            other => panic!("expected transformed censored observation, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_source_unit_abstains_without_fabricating_a_conversion() {
        let source = mapped_source_frame(SensorObservationStateV1::Observed {
            value: 2.5,
            unit: "bar".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.02,
                coverage_probability: Some(0.95),
                method_ref: "gauge-report-4".into(),
            },
        });
        let execution =
            execute_sensor_correspondence_v1(&source, &complete_correspondence_manifest())
                .unwrap();
        assert!(execution.report.abstain_required);
        assert!(execution.target_frame.is_none());
        assert!(
            execution
                .report
                .abstention_reasons
                .iter()
                .any(|reason| reason.contains("differs from declared transform source unit"))
        );
    }

    #[test]
    fn correspondence_order_does_not_change_canonical_target_frame() {
        let source = mapped_source_frame(SensorObservationStateV1::Observed {
            value: 2.5,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.02,
                coverage_probability: Some(0.95),
                method_ref: "gauge-report-4".into(),
            },
        });
        let manifest = complete_correspondence_manifest();
        let mut reordered = manifest.clone();
        reordered.correspondences.reverse();

        let first = execute_sensor_correspondence_v1(&source, &manifest).unwrap();
        let second = execute_sensor_correspondence_v1(&source, &reordered).unwrap();
        assert_eq!(first.target_frame, second.target_frame);
        assert_eq!(
            first.report.correspondence_manifest_sha256,
            second.report.correspondence_manifest_sha256
        );
        assert_eq!(first.report.target_frame_sha256, second.report.target_frame_sha256);
        assert_eq!(first.report.report_sha256, second.report.report_sha256);
    }

    #[test]
    fn unmapped_source_sensor_is_reported_not_positionally_reused() {
        let mut source = mapped_source_frame(SensorObservationStateV1::Observed {
            value: 2.5,
            unit: "kPa".into(),
            uncertainty: MeasurementUncertaintyV1 {
                standard_uncertainty: 0.02,
                coverage_probability: Some(0.95),
                method_ref: "gauge-report-4".into(),
            },
        });
        source.observations.push(SensorObservationV1 {
            sensor_id: "source-extra-temperature".into(),
            state: SensorObservationStateV1::ObservedUnqualified { value: 300.0 },
            instrument_id: None,
            calibration_ref: None,
        });
        let execution =
            execute_sensor_correspondence_v1(&source, &complete_correspondence_manifest())
                .unwrap();
        assert!(!execution.report.abstain_required);
        assert_eq!(
            execution.report.unmapped_source_sensor_ids,
            vec!["source-extra-temperature".to_string()]
        );
        assert_eq!(
            execution
                .target_frame
                .as_ref()
                .unwrap()
                .observations
                .len(),
            2
        );
    }

    #[test]
    fn source_schema_digest_must_match_correspondence_manifest() {
        let source = mapped_source_frame(SensorObservationStateV1::UnknownMissingReason);
        let mut manifest = complete_correspondence_manifest();
        manifest.source_sensor_schema_sha256 = SHA_C.into();
        assert!(execute_sensor_correspondence_v1(&source, &manifest).is_err());
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
