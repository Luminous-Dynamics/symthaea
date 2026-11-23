// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable consumed-label manifests and K0/K1 snapshot requalification.
//!
//! This module records the exact set of qualified metrology labels used by a
//! VM training/evaluation snapshot. It replays per-label eligibility at the
//! historical K0 frontier and compares that same fixed set with current K1
//! eligibility. It never edits labels, training manifests, model snapshots,
//! or model parameters; it does not execute training or process control.

use chrono::{DateTime, FixedOffset, NaiveDateTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::calibration_requalification::CalibrationRequalificationInputV1;
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
    assess_label_consumption_eligibility_v1,
};
use super::process_event_graph::{ClockBasisV1, MetrologyLabelV1, PredictionInformationCutoffV1};

pub const VM_SNAPSHOT_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.vm-snapshot-input.v1";
pub const VM_SNAPSHOT_MANIFEST_SCHEMA_V1: &str = "luminous.semi-bench.vm-snapshot-manifest.v1";
pub const VM_SNAPSHOT_REQUALIFICATION_SCHEMA_V1: &str =
    "luminous.semi-bench.vm-snapshot-requalification.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmSnapshotPurposeV1 {
    Training,
    Evaluation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmSnapshotCurrentDispositionV1 {
    CurrentEligible,
    RebuildRequired,
    Indeterminate,
    HistoricalOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotLabelEvidenceV1 {
    pub label: MetrologyLabelV1,
    pub calibration_input: CalibrationRequalificationInputV1,
    /// Required only where the label entered through a metrology audit policy.
    pub selection_receipt_required: bool,
    pub selection_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotInputV1 {
    pub schema: String,
    pub input_id: String,
    pub snapshot_id: String,
    pub purpose: VmSnapshotPurposeV1,
    pub model_artifact_sha256: String,
    pub model_config_sha256: String,
    pub split_manifest_sha256: String,
    pub preprocessing_sha256: String,
    pub cutoff: PredictionInformationCutoffV1,
    /// Every row consumed by the snapshot must appear exactly once.
    pub consumed_labels: Vec<VmSnapshotLabelEvidenceV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotLabelManifestEntryV1 {
    pub label_id: String,
    pub label_revision: u64,
    pub subject_id: String,
    pub measurand_id: String,
    pub measurement_event_id: String,
    pub label_source_artifact_sha256: String,
    pub label_payload_sha256: String,
    pub measurement_sha256: String,
    pub qualification_report_sha256: String,
    pub calibration_certificate_id: String,
    pub calibration_certificate_sha256: String,
    pub k0_frontier_sha256: String,
    /// Stable K0-only projection; deliberately excludes the combined K0/K1 report digest.
    pub k0_eligibility_projection_sha256: String,
    pub k0_eligibility: LabelConsumptionEligibilityV1,
    pub selection_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmTrainingSnapshotManifestV1 {
    pub schema: String,
    pub snapshot_id: String,
    pub purpose: VmSnapshotPurposeV1,
    pub model_artifact_sha256: String,
    pub model_config_sha256: String,
    pub split_manifest_sha256: String,
    pub preprocessing_sha256: String,
    pub cutoff: PredictionInformationCutoffV1,
    pub k0_unix_seconds: i64,
    pub consumed_label_count: usize,
    pub entries: Vec<VmSnapshotLabelManifestEntryV1>,
    pub source_authenticity_proven: bool,
    pub training_snapshot_mutated: bool,
    pub equipment_control_authority: bool,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotLabelCurrentStatusV1 {
    pub label_id: String,
    pub k0_projection_sha256: String,
    pub k1_eligibility: LabelConsumptionEligibilityV1,
    pub k1_report_sha256: String,
    pub reasons: Vec<super::label_consumption_guard::LabelConsumptionReasonV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotRequalificationReportV1 {
    pub schema: String,
    pub snapshot_id: String,
    pub historical_manifest_sha256: String,
    pub requalified_at_k1_unix_seconds: i64,
    pub current_disposition: VmSnapshotCurrentDispositionV1,
    pub consumed_label_count: usize,
    pub current_eligible_count: usize,
    pub current_unknown_count: usize,
    pub current_blocked_count: usize,
    pub label_statuses: Vec<VmSnapshotLabelCurrentStatusV1>,
    /// Historical identity is reported unchanged; K1 creates a separate receipt.
    pub historical_manifest_rewritten: bool,
    pub training_snapshot_mutated: bool,
    pub model_retrained: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct VmSnapshotEvidenceErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), VmSnapshotEvidenceErrorV1> {
    if value.trim().is_empty() {
        return Err(VmSnapshotEvidenceErrorV1(format!(
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, VmSnapshotEvidenceErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        VmSnapshotEvidenceErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn manifest_digest(
    manifest: &VmTrainingSnapshotManifestV1,
) -> Result<String, VmSnapshotEvidenceErrorV1> {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    json_sha256(&canonical)
}

fn report_digest(
    report: &VmSnapshotRequalificationReportV1,
) -> Result<String, VmSnapshotEvidenceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn timestamp_micros(
    value: &str,
    basis: &ClockBasisV1,
    field: &str,
) -> Result<i64, VmSnapshotEvidenceErrorV1> {
    match basis {
        ClockBasisV1::OffsetAware => {
            let parsed = DateTime::<FixedOffset>::parse_from_rfc3339(value).map_err(|_| {
                VmSnapshotEvidenceErrorV1(format!(
                    "{field}: expected RFC3339 timestamp with an explicit offset"
                ))
            })?;
            Ok(parsed.timestamp_micros())
        }
        ClockBasisV1::SourceLocalNaive => {
            let formats = [
                "%Y-%m-%dT%H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S%.f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ];
            for format in formats {
                if let Ok(parsed) = NaiveDateTime::parse_from_str(value, format) {
                    // Comparable only inside this declared source-local clock domain.
                    return Ok(parsed.and_utc().timestamp_micros());
                }
            }
            Err(VmSnapshotEvidenceErrorV1(format!(
                "{field}: invalid source-local naive timestamp"
            )))
        }
    }
}

fn k0_projection_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, VmSnapshotEvidenceErrorV1> {
    let mut historical = report.clone();
    // The combined report hash binds K1 as well as K0. The K0 frontier digest
    // and the other fields below are the historical projection.
    historical.requalification_report_sha256.clear();
    historical.report_sha256.clear();
    json_sha256(&historical)
}

fn validate_input_shape(input: &VmSnapshotInputV1) -> Result<(), VmSnapshotEvidenceErrorV1> {
    if input.schema != VM_SNAPSHOT_INPUT_SCHEMA_V1 {
        return Err(VmSnapshotEvidenceErrorV1(
            "input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    require_text(&input.snapshot_id, "input.snapshot_id")?;
    require_text(&input.cutoff.cutoff_id, "input.cutoff.cutoff_id")?;
    require_text(
        &input.cutoff.clock_domain_id,
        "input.cutoff.clock_domain_id",
    )?;

    for (name, value) in [
        (
            "model_artifact_sha256",
            input.model_artifact_sha256.as_str(),
        ),
        ("model_config_sha256", input.model_config_sha256.as_str()),
        (
            "split_manifest_sha256",
            input.split_manifest_sha256.as_str(),
        ),
        ("preprocessing_sha256", input.preprocessing_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "input.{name}: malformed lowercase SHA-256 digest"
            )));
        }
    }
    if input.consumed_labels.is_empty() {
        return Err(VmSnapshotEvidenceErrorV1(
            "a VM snapshot must declare at least one consumed label".into(),
        ));
    }
    Ok(())
}

fn validate_label_time(
    label: &MetrologyLabelV1,
    cutoff: &PredictionInformationCutoffV1,
) -> Result<(), VmSnapshotEvidenceErrorV1> {
    if !matches!(&cutoff.clock_basis, ClockBasisV1::OffsetAware) {
        return Err(VmSnapshotEvidenceErrorV1(
            "source-local timestamps require an explicit clock mapping before comparison with Unix calibration bounds"
                .into(),
        ));
    }
    if label.measurement_clock_domain_id != cutoff.clock_domain_id
        || label.availability_clock_domain_id != cutoff.clock_domain_id
        || label.measurement_clock_basis != cutoff.clock_basis
        || label.availability_clock_basis != cutoff.clock_basis
    {
        return Err(VmSnapshotEvidenceErrorV1(format!(
            "label {} has clock-domain or clock-basis mismatch with snapshot cutoff",
            label.label_id
        )));
    }

    let cutoff_micros = timestamp_micros(
        &cutoff.cutoff_time,
        &cutoff.clock_basis,
        "cutoff.cutoff_time",
    )?;
    let measured_micros = timestamp_micros(
        &label.measurement_time,
        &label.measurement_clock_basis,
        "label.measurement_time",
    )?;
    let available_micros = timestamp_micros(
        &label.available_at,
        &label.availability_clock_basis,
        "label.available_at",
    )?;
    if measured_micros > cutoff_micros {
        return Err(VmSnapshotEvidenceErrorV1(format!(
            "label {} measurement occurs after the snapshot information cutoff",
            label.label_id
        )));
    }
    if available_micros > cutoff_micros {
        return Err(VmSnapshotEvidenceErrorV1(format!(
            "label {} was not available at the snapshot information cutoff",
            label.label_id
        )));
    }
    Ok(())
}

fn label_measurement_time_micros(
    label: &MetrologyLabelV1,
) -> Result<i64, VmSnapshotEvidenceErrorV1> {
    timestamp_micros(
        &label.measurement_time,
        &label.measurement_clock_basis,
        "label.measurement_time",
    )
}

fn validate_label_identity(label: &MetrologyLabelV1) -> Result<(), VmSnapshotEvidenceErrorV1> {
    require_text(&label.label_id, "label.label_id")?;
    require_text(&label.measurement_event_id, "label.measurement_event_id")?;
    require_text(&label.subject_id, "label.subject_id")?;
    require_text(&label.measurand_id, "label.measurand_id")?;
    require_text(&label.unit, "label.unit")?;
    require_text(&label.authority_ref, "label.authority_ref")?;
    if label.revision == 0 || !valid_sha256(&label.source_artifact_sha256) {
        return Err(VmSnapshotEvidenceErrorV1(format!(
            "label {} has invalid revision or source artifact digest",
            label.label_id
        )));
    }
    Ok(())
}

/// Builds an immutable training/evaluation manifest from the exact consumed-label set.
///
/// Every label must be eligible at the snapshot's K0 frontier. The function
/// refuses to silently drop a label, and rejects duplicate label/event/subject
/// identities or a snapshot that includes both a label and its superseded row.
pub fn build_vm_training_snapshot_manifest_v1(
    input: &VmSnapshotInputV1,
) -> Result<VmTrainingSnapshotManifestV1, VmSnapshotEvidenceErrorV1> {
    validate_input_shape(input)?;

    let mut label_ids = BTreeSet::new();
    let mut measurement_event_ids = BTreeSet::new();
    let mut subject_measurands = BTreeSet::new();
    let mut superseded_ids = BTreeSet::new();
    let mut entries = Vec::with_capacity(input.consumed_labels.len());
    let mut common_k0: Option<i64> = None;
    let mut common_k1: Option<i64> = None;

    let cutoff_micros = timestamp_micros(
        &input.cutoff.cutoff_time,
        &input.cutoff.clock_basis,
        "cutoff.cutoff_time",
    )?;
    for packet in &input.consumed_labels {
        let label = &packet.label;
        validate_label_identity(label)?;
        validate_label_time(label, &input.cutoff)?;

        if !label_ids.insert(label.label_id.as_str()) {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "duplicate consumed label ID {}",
                label.label_id
            )));
        }
        if !measurement_event_ids.insert(label.measurement_event_id.as_str()) {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "duplicate consumed measurement event {}",
                label.measurement_event_id
            )));
        }
        let subject_measurand = (label.subject_id.as_str(), label.measurand_id.as_str());
        if !subject_measurands.insert(subject_measurand) {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "duplicate active label for subject {} and measurand {}",
                label.subject_id, label.measurand_id
            )));
        }
        if let Some(superseded) = label.supersedes_label_id.as_deref() {
            require_text(superseded, "label.supersedes_label_id")?;
            if !superseded_ids.insert(superseded) {
                return Err(VmSnapshotEvidenceErrorV1(format!(
                    "multiple labels supersede the same predecessor {superseded}"
                )));
            }
        }

        let k0 = packet.calibration_input.k0_unix_seconds;
        if common_k0.is_some_and(|existing| existing != k0) {
            return Err(VmSnapshotEvidenceErrorV1(
                "consumed labels do not share one K0 calibration frontier".into(),
            ));
        }
        common_k0 = Some(k0);
        let k1 = packet.calibration_input.k1_unix_seconds;
        if common_k1.is_some_and(|existing| existing != k1) {
            return Err(VmSnapshotEvidenceErrorV1(
                "consumed labels do not share one K1 calibration frontier".into(),
            ));
        }
        common_k1 = Some(k1);
        if matches!(input.cutoff.clock_basis, ClockBasisV1::OffsetAware)
            && cutoff_micros / 1_000_000 != k0
        {
            return Err(VmSnapshotEvidenceErrorV1(
                "snapshot cutoff and per-label K0 calibration frontier disagree".into(),
            ));
        }
        let measurement_time_micros = label_measurement_time_micros(label)?;
        let measured_lower = packet
            .calibration_input
            .measurement_time_bounds
            .earliest_measured_at_unix_seconds
            .checked_mul(1_000_000)
            .ok_or_else(|| VmSnapshotEvidenceErrorV1("measurement lower time overflows".into()))?;
        let measured_upper = packet
            .calibration_input
            .measurement_time_bounds
            .latest_measured_at_unix_seconds
            .checked_mul(1_000_000)
            .ok_or_else(|| VmSnapshotEvidenceErrorV1("measurement upper time overflows".into()))?;
        if measurement_time_micros < measured_lower || measurement_time_micros > measured_upper {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "label {} measurement time is outside its calibration evidence time bounds",
                label.label_id
            )));
        }

        if packet.calibration_input.measurement.measurement_event_id != label.measurement_event_id {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "label {} does not bind to its calibration evidence measurement event",
                label.label_id
            )));
        }
        if packet.selection_receipt_required && packet.selection_receipt_sha256.is_none() {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "label {} requires a metrology selection receipt but none was supplied",
                label.label_id
            )));
        }
        if let Some(selection_digest) = packet.selection_receipt_sha256.as_deref()
            && !valid_sha256(selection_digest)
        {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "label {} has malformed selection receipt digest",
                label.label_id
            )));
        }

        let report = assess_label_consumption_eligibility_v1(
            label,
            &packet.calibration_input,
            LabelEvidenceFrontierV1::AsKnownAtK0,
        )
        .map_err(|error| VmSnapshotEvidenceErrorV1(error.to_string()))?;
        if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            || !report.downstream_consumption_allowed
        {
            return Err(VmSnapshotEvidenceErrorV1(format!(
                "label {} is not eligible at K0: {:?}",
                label.label_id, report.reasons
            )));
        }

        let projection = k0_projection_digest(&report)?;
        entries.push(VmSnapshotLabelManifestEntryV1 {
            label_id: label.label_id.clone(),
            label_revision: label.revision,
            subject_id: label.subject_id.clone(),
            measurand_id: label.measurand_id.clone(),
            measurement_event_id: label.measurement_event_id.clone(),
            label_source_artifact_sha256: label.source_artifact_sha256.clone(),
            label_payload_sha256: report.label_sha256,
            measurement_sha256: report.measurement_sha256,
            qualification_report_sha256: report.qualification_report_sha256,
            calibration_certificate_id: report.calibration_certificate_id,
            calibration_certificate_sha256: packet
                .calibration_input
                .calibration_at_k0
                .certificate_sha256
                .clone(),
            k0_frontier_sha256: report.requalification_frontier_sha256,
            k0_eligibility_projection_sha256: projection,
            k0_eligibility: report.eligibility,
            selection_receipt_sha256: packet.selection_receipt_sha256.clone(),
        });
    }

    if superseded_ids.iter().any(|id| label_ids.contains(*id)) {
        return Err(VmSnapshotEvidenceErrorV1(
            "snapshot contains a label and its explicitly superseded predecessor".into(),
        ));
    }

    entries.sort_by(|left, right| left.label_id.cmp(&right.label_id));
    let mut manifest = VmTrainingSnapshotManifestV1 {
        schema: VM_SNAPSHOT_MANIFEST_SCHEMA_V1.into(),
        snapshot_id: input.snapshot_id.clone(),
        purpose: input.purpose,
        model_artifact_sha256: input.model_artifact_sha256.clone(),
        model_config_sha256: input.model_config_sha256.clone(),
        split_manifest_sha256: input.split_manifest_sha256.clone(),
        preprocessing_sha256: input.preprocessing_sha256.clone(),
        cutoff: input.cutoff.clone(),
        k0_unix_seconds: common_k0.expect("non-empty consumed labels were validated"),
        consumed_label_count: entries.len(),
        entries,
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    Ok(manifest)
}

/// Requalifies the exact historical consumed-label set at K1 without rewriting K0.
///
/// The caller must provide the same frozen label/evidence packet set used to
/// build the manifest, except that calibration inputs may now include later
/// owner events and K1 materialization. The K0-only projection is replayed and
/// compared before K1 status is assessed.
pub fn requalify_vm_training_snapshot_v1(
    input: &VmSnapshotInputV1,
    historical_manifest: &VmTrainingSnapshotManifestV1,
) -> Result<VmSnapshotRequalificationReportV1, VmSnapshotEvidenceErrorV1> {
    if manifest_digest(historical_manifest)? != historical_manifest.manifest_sha256 {
        return Err(VmSnapshotEvidenceErrorV1(
            "historical manifest content does not match its declared digest".into(),
        ));
    }
    let replayed_manifest = build_vm_training_snapshot_manifest_v1(input)?;
    if replayed_manifest.manifest_sha256 != historical_manifest.manifest_sha256 {
        return Err(VmSnapshotEvidenceErrorV1(
            "historical K0 snapshot manifest does not replay byte-identically".into(),
        ));
    }

    let mut label_statuses = Vec::with_capacity(input.consumed_labels.len());
    let mut current_eligible_count = 0;
    let mut current_unknown_count = 0;
    let mut current_blocked_count = 0;

    for packet in &input.consumed_labels {
        let entry = historical_manifest
            .entries
            .iter()
            .find(|entry| entry.label_id == packet.label.label_id)
            .ok_or_else(|| {
                VmSnapshotEvidenceErrorV1(format!(
                    "historical manifest has no entry for label {}",
                    packet.label.label_id
                ))
            })?;
        let k1_report = assess_label_consumption_eligibility_v1(
            &packet.label,
            &packet.calibration_input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .map_err(|error| VmSnapshotEvidenceErrorV1(error.to_string()))?;

        match k1_report.eligibility {
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
                current_eligible_count += 1;
            }
            LabelConsumptionEligibilityV1::Unknown => current_unknown_count += 1,
            LabelConsumptionEligibilityV1::Blocked => current_blocked_count += 1,
        }
        label_statuses.push(VmSnapshotLabelCurrentStatusV1 {
            label_id: packet.label.label_id.clone(),
            k0_projection_sha256: entry.k0_eligibility_projection_sha256.clone(),
            k1_eligibility: k1_report.eligibility,
            k1_report_sha256: k1_report.report_sha256,
            reasons: k1_report.reasons,
        });
    }
    label_statuses.sort_by(|left, right| left.label_id.cmp(&right.label_id));

    let current_disposition = if current_blocked_count > 0 {
        match historical_manifest.purpose {
            VmSnapshotPurposeV1::Training => VmSnapshotCurrentDispositionV1::RebuildRequired,
            VmSnapshotPurposeV1::Evaluation => VmSnapshotCurrentDispositionV1::HistoricalOnly,
        }
    } else if current_unknown_count > 0 {
        VmSnapshotCurrentDispositionV1::Indeterminate
    } else {
        VmSnapshotCurrentDispositionV1::CurrentEligible
    };

    let requalified_at = input
        .consumed_labels
        .iter()
        .map(|packet| packet.calibration_input.k1_unix_seconds)
        .max()
        .unwrap_or(historical_manifest.k0_unix_seconds);

    let mut report = VmSnapshotRequalificationReportV1 {
        schema: VM_SNAPSHOT_REQUALIFICATION_SCHEMA_V1.into(),
        snapshot_id: historical_manifest.snapshot_id.clone(),
        historical_manifest_sha256: historical_manifest.manifest_sha256.clone(),
        requalified_at_k1_unix_seconds: requalified_at,
        current_disposition,
        consumed_label_count: historical_manifest.consumed_label_count,
        current_eligible_count,
        current_unknown_count,
        current_blocked_count,
        label_statuses,
        historical_manifest_rewritten: false,
        training_snapshot_mutated: false,
        model_retrained: false,
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
        CalibrationOperatingRangeV1, MeasurementTimeBoundsV1, seal_calibration_event_inventory_v1,
    };
    use crate::benchmark::metrology_qualification_gate::{
        CALIBRATION_CERTIFICATE_SCHEMA_V1, CalibrationCertificateV1,
        MEASUREMENT_EVIDENCE_SCHEMA_V1, MeasurementEvidenceV1, MeasurementQualificationPolicyV1,
        MeasurementUncertaintyComponentV1, QUALIFICATION_POLICY_SCHEMA_V1,
        TraceabilityDeclarationV1, UncertaintyComponentKindV1, qualify_measurement_evidence_v1,
    };
    use crate::benchmark::process_event_graph::MetrologyLabelStatusV1;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

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
            certificate_sha256: digest('a'),
            issuing_authority_ref: "metrology-lab-01".into(),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            reference_standard_id: "step-height-std-02".into(),
            reference_certificate_sha256: Some(digest('b')),
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

    fn measurement(event_id: &str, source_digest: &str) -> MeasurementEvidenceV1 {
        MeasurementEvidenceV1 {
            schema: MEASUREMENT_EVIDENCE_SCHEMA_V1.into(),
            measurement_event_id: event_id.into(),
            source_record_sha256: source_digest.into(),
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
                source_sha256: digest('d'),
                kind: UncertaintyComponentKindV1::TypeA,
                standard_uncertainty: 0.03,
            }],
        }
    }

    fn calibration_input(
        event_id: &str,
        event_source_digest: &str,
    ) -> CalibrationRequalificationInputV1 {
        let measured = measurement(event_id, event_source_digest);
        let calibrated = certificate();
        let qualification_policy = policy();
        let qualification =
            qualify_measurement_evidence_v1(&measured, Some(&calibrated), &qualification_policy)
                .unwrap();
        CalibrationRequalificationInputV1 {
            schema: CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1.into(),
            input_id: format!("cal-input-{event_id}"),
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

    fn label(label_id: &str, event_id: &str, subject: &str) -> MetrologyLabelV1 {
        MetrologyLabelV1 {
            label_id: label_id.into(),
            measurement_event_id: event_id.into(),
            subject_id: subject.into(),
            measurand_id: "film-thickness".into(),
            value: Some(12.45),
            unit: "nm".into(),
            standard_uncertainty: Some(0.09),
            censoring_lower_bound: None,
            censoring_upper_bound: None,
            censoring_basis: None,
            status: MetrologyLabelStatusV1::Validated,
            measurement_time: "2028-08-16T00:53:20Z".into(),
            measurement_clock_domain_id: "fab-clock".into(),
            measurement_clock_basis: ClockBasisV1::OffsetAware,
            available_at: "2028-08-16T00:58:20Z".into(),
            availability_clock_domain_id: "fab-clock".into(),
            availability_clock_basis: ClockBasisV1::OffsetAware,
            instrument_id: Some("ellipsometer-07".into()),
            calibration_ref: Some("cal-cert-001".into()),
            revision: 1,
            supersedes_label_id: None,
            source_artifact_sha256: digest('f'),
            authority_ref: "metrology-owner-01".into(),
        }
    }

    fn packet(label: MetrologyLabelV1) -> VmSnapshotLabelEvidenceV1 {
        VmSnapshotLabelEvidenceV1 {
            calibration_input: calibration_input(&label.measurement_event_id, &digest('c')),
            label,
            selection_receipt_required: true,
            selection_receipt_sha256: Some(digest('e')),
        }
    }

    fn snapshot_input() -> VmSnapshotInputV1 {
        VmSnapshotInputV1 {
            schema: VM_SNAPSHOT_INPUT_SCHEMA_V1.into(),
            input_id: "vm-snapshot-input-001".into(),
            snapshot_id: "vm-snapshot-001".into(),
            purpose: VmSnapshotPurposeV1::Training,
            model_artifact_sha256: digest('1'),
            model_config_sha256: digest('2'),
            split_manifest_sha256: digest('3'),
            preprocessing_sha256: digest('4'),
            cutoff: PredictionInformationCutoffV1 {
                cutoff_id: "fab-k0-cutoff".into(),
                cutoff_time: "2028-12-09T18:40:00Z".into(),
                clock_domain_id: "fab-clock".into(),
                clock_basis: ClockBasisV1::OffsetAware,
            },
            consumed_labels: vec![packet(label("label-001", "measurement-001", "wafer-001"))],
        }
    }

    fn add_revocation(input: &mut VmSnapshotInputV1, materialize: bool) {
        for packet in &mut input.consumed_labels {
            let event = CalibrationChangeEventV1 {
                schema: CALIBRATION_CHANGE_EVENT_SCHEMA_V1.into(),
                event_id: format!("revoke-{}", packet.label.label_id),
                event_kind: CalibrationChangeKindV1::Revoked,
                source_id: "metrology-lab-01".into(),
                source_payload_sha256: digest('9'),
                authority_ref: "calibration-owner-01".into(),
                source_sequence: Some(1),
                effective_at_unix_seconds: 1_840_000_000,
                observed_at_unix_seconds: 1_870_000_000,
                qualification: CalibrationEventQualificationV1::Qualified,
                scope: Some(CalibrationChangeScopeV1 {
                    target: CalibrationChangeTargetV1::Certificate {
                        certificate_id: "cal-cert-001".into(),
                        certificate_sha256: digest('a'),
                    },
                    instrument_id: "ellipsometer-07".into(),
                    measurand_id: "film-thickness".into(),
                    unit: "nm".into(),
                    method_revision: "film-method-3".into(),
                    operating_range: CalibrationOperatingRangeV1::EntireDeclaredCertificateRange,
                }),
            };
            packet.calibration_input.owner_event_inventory_sha256 =
                seal_calibration_event_inventory_v1(std::slice::from_ref(&event)).unwrap();
            packet.calibration_input.owner_events = vec![event.clone()];
            if materialize {
                packet.calibration_input.k1.materialized_event_ids = vec![event.event_id];
            }
        }
    }

    fn assert_no_mutation(report: &VmSnapshotRequalificationReportV1) {
        assert!(!report.historical_manifest_rewritten);
        assert!(!report.training_snapshot_mutated);
        assert!(!report.model_retrained);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn fully_qualified_consumed_set_is_currently_eligible() {
        let input = snapshot_input();
        let manifest = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        let report = requalify_vm_training_snapshot_v1(&input, &manifest).unwrap();
        assert_eq!(manifest.consumed_label_count, 1);
        assert_eq!(
            report.current_disposition,
            VmSnapshotCurrentDispositionV1::CurrentEligible
        );
        assert_eq!(report.current_eligible_count, 1);
        assert_no_mutation(&report);
    }

    #[test]
    fn retroactive_revocation_preserves_k0_manifest_and_requires_rebuild() {
        let mut input = snapshot_input();
        let historical = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        let historical_digest = historical.manifest_sha256.clone();
        add_revocation(&mut input, true);

        let replayed = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        assert_eq!(replayed.manifest_sha256, historical_digest);
        let report = requalify_vm_training_snapshot_v1(&input, &historical).unwrap();
        assert_eq!(
            report.current_disposition,
            VmSnapshotCurrentDispositionV1::RebuildRequired
        );
        assert_eq!(report.current_blocked_count, 1);
        assert_eq!(report.historical_manifest_sha256, historical_digest);
        assert_no_mutation(&report);
    }

    #[test]
    fn unmaterialized_relevant_revocation_is_indeterminate_not_eligible() {
        let mut input = snapshot_input();
        let historical = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        add_revocation(&mut input, false);
        let report = requalify_vm_training_snapshot_v1(&input, &historical).unwrap();
        assert_eq!(
            report.current_disposition,
            VmSnapshotCurrentDispositionV1::Indeterminate
        );
        assert_eq!(report.current_unknown_count, 1);
        assert_eq!(report.current_eligible_count, 0);
    }

    #[test]
    fn post_cutoff_label_availability_is_rejected() {
        let mut input = snapshot_input();
        input.consumed_labels[0].label.available_at = "2028-12-09T18:45:00Z".into();
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn post_cutoff_measurement_time_is_rejected() {
        let mut input = snapshot_input();
        input.consumed_labels[0].label.measurement_time = "2026-11-20T10:15:00Z".into();
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn duplicate_label_and_measurement_event_ids_are_rejected() {
        let mut input = snapshot_input();
        input
            .consumed_labels
            .push(packet(label("label-001", "measurement-002", "wafer-002")));
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());

        let mut input = snapshot_input();
        input
            .consumed_labels
            .push(packet(label("label-002", "measurement-001", "wafer-002")));
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn two_active_labels_for_same_subject_and_measurand_are_rejected() {
        let mut input = snapshot_input();
        input
            .consumed_labels
            .push(packet(label("label-002", "measurement-002", "wafer-001")));
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn superseded_predecessor_cannot_be_consumed_with_its_successor() {
        let mut input = snapshot_input();
        let mut successor = label("label-002", "measurement-002", "wafer-002");
        successor.supersedes_label_id = Some("label-001".into());
        input.consumed_labels.push(packet(successor));
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn required_but_missing_selection_receipt_is_rejected() {
        let mut input = snapshot_input();
        input.consumed_labels[0].selection_receipt_sha256 = None;
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn source_local_naive_clock_is_not_compared_to_unix_calibration_time() {
        let mut input = snapshot_input();
        input.cutoff.clock_basis = ClockBasisV1::SourceLocalNaive;
        input.consumed_labels[0].label.measurement_clock_basis = ClockBasisV1::SourceLocalNaive;
        input.consumed_labels[0].label.availability_clock_basis = ClockBasisV1::SourceLocalNaive;
        input.cutoff.cutoff_time = "2028-12-09T18:40:00".into();
        input.consumed_labels[0].label.measurement_time = "2028-08-16T00:53:20".into();
        input.consumed_labels[0].label.available_at = "2028-08-16T00:58:20".into();
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn source_clock_mismatch_and_malformed_selection_receipt_are_rejected() {
        let mut input = snapshot_input();
        input.consumed_labels[0].selection_receipt_sha256 = Some("invalid".into());
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());

        let mut input = snapshot_input();
        input.consumed_labels[0].label.availability_clock_domain_id = "other-clock".into();
        assert!(build_vm_training_snapshot_manifest_v1(&input).is_err());
    }

    #[test]
    fn entry_order_does_not_change_manifest_digest() {
        let mut input = snapshot_input();
        input
            .consumed_labels
            .push(packet(label("label-002", "measurement-002", "wafer-002")));
        let first = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        input.consumed_labels.reverse();
        let second = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
    }

    #[test]
    fn numeric_label_metadata_is_explicitly_bound_into_snapshot_manifest() {
        let original = snapshot_input();
        let first = build_vm_training_snapshot_manifest_v1(&original).unwrap();

        let mut changed = snapshot_input();
        changed.consumed_labels[0].label.standard_uncertainty = Some(0.10);
        let second = build_vm_training_snapshot_manifest_v1(&changed).unwrap();

        assert_ne!(
            first.entries[0].label_payload_sha256,
            second.entries[0].label_payload_sha256
        );
        assert_ne!(first.manifest_sha256, second.manifest_sha256);
    }

    #[test]
    fn model_or_preprocessing_change_changes_manifest_digest() {
        let input = snapshot_input();
        let first = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        let mut changed = input;
        changed.preprocessing_sha256 = digest('5');
        let second = build_vm_training_snapshot_manifest_v1(&changed).unwrap();
        assert_ne!(first.manifest_sha256, second.manifest_sha256);
    }

    #[test]
    fn stale_manifest_digest_is_rejected_even_if_snapshot_id_matches() {
        let input = snapshot_input();
        let mut manifest = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        manifest.entries[0].label_revision += 1;
        assert!(requalify_vm_training_snapshot_v1(&input, &manifest).is_err());
    }

    #[test]
    fn historical_manifest_tampering_is_rejected_on_replay() {
        let input = snapshot_input();
        let mut manifest = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        manifest.entries[0].label_revision += 1;
        assert!(requalify_vm_training_snapshot_v1(&input, &manifest).is_err());
    }

    #[test]
    fn evaluation_snapshot_with_a_blocked_label_is_historical_only() {
        let mut input = snapshot_input();
        input.purpose = VmSnapshotPurposeV1::Evaluation;
        let manifest = build_vm_training_snapshot_manifest_v1(&input).unwrap();
        add_revocation(&mut input, true);
        let report = requalify_vm_training_snapshot_v1(&input, &manifest).unwrap();
        assert_eq!(
            report.current_disposition,
            VmSnapshotCurrentDispositionV1::HistoricalOnly
        );
    }
}
