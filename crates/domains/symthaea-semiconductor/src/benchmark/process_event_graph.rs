// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-backed process-event graph and delayed metrology-label lifecycle.
//!
//! Event time and result-availability time are separate. A label cannot be
//! consumed before it is available in the information stream, regardless of
//! when the physical measurement occurred.

use chrono::{DateTime, FixedOffset, NaiveDateTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const GRAPH_SCHEMA_V1: &str = "luminous.semi-bench.process-event-graph.v1";
pub const SNAPSHOT_SCHEMA_V1: &str = "luminous.semi-bench.information-cutoff-snapshot.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockBasisV1 {
    OffsetAware,
    SourceLocalNaive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessEventKindV1 {
    Load,
    RecipeStart,
    ProcessStepStart,
    ProcessStepEnd,
    RecipeEnd,
    Unload,
    Maintenance,
    Calibration,
    Alarm,
    Metrology,
    ElectricalTest,
    Disposition,
    Other,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventV1 {
    pub event_id: String,
    pub event_kind: ProcessEventKindV1,
    /// Subject/wafer/lot identities are nullable and never fabricated.
    pub subject_id: Option<String>,
    pub wafer_id: Option<String>,
    pub lot_id: Option<String>,
    pub carrier_id: Option<String>,
    pub tool_id: Option<String>,
    pub chamber_id: Option<String>,
    pub module_id: Option<String>,
    pub process_step_id: Option<String>,
    pub recipe_id: Option<String>,
    pub recipe_revision: Option<String>,
    pub event_time: String,
    pub event_clock_domain_id: String,
    pub event_clock_basis: ClockBasisV1,
    /// Earliest time this event became visible to the consuming data system.
    pub available_at: String,
    pub availability_clock_domain_id: String,
    pub availability_clock_basis: ClockBasisV1,
    pub source_sequence: Option<u64>,
    pub source_artifact_sha256: String,
    pub transform_sha256: String,
    pub authority_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessEdgeKindV1 {
    Precedes,
    Measures,
    SameSubjectSequence,
    MaintenanceContext,
    CalibrationContext,
    RecipeContext,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventEdgeV1 {
    pub edge_id: String,
    pub source_event_id: String,
    pub target_event_id: String,
    pub edge_kind: ProcessEdgeKindV1,
    /// Evidence-backed ordering/relationship, not inferred physical causality.
    pub evidence_ref: String,
    pub source_artifact_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyLabelStatusV1 {
    Validated,
    Provisional,
    Censored,
    Invalid,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyLabelV1 {
    pub label_id: String,
    pub measurement_event_id: String,
    pub subject_id: String,
    pub measurand_id: String,
    pub value: Option<f64>,
    pub unit: String,
    pub standard_uncertainty: Option<f64>,
    pub censoring_lower_bound: Option<f64>,
    pub censoring_upper_bound: Option<f64>,
    pub censoring_basis: Option<String>,
    pub status: MetrologyLabelStatusV1,
    pub measurement_time: String,
    pub measurement_clock_domain_id: String,
    pub measurement_clock_basis: ClockBasisV1,
    /// Earliest time the result could be consumed by training/evaluation.
    pub available_at: String,
    pub availability_clock_domain_id: String,
    pub availability_clock_basis: ClockBasisV1,
    pub instrument_id: Option<String>,
    pub calibration_ref: Option<String>,
    pub revision: u64,
    pub supersedes_label_id: Option<String>,
    pub source_artifact_sha256: String,
    pub authority_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventGraphV1 {
    pub schema: String,
    pub graph_id: String,
    pub source_artifact_sha256: String,
    pub transform_sha256: String,
    pub events: Vec<ProcessEventV1>,
    pub edges: Vec<ProcessEventEdgeV1>,
    pub metrology_labels: Vec<MetrologyLabelV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionInformationCutoffV1 {
    pub cutoff_id: String,
    pub cutoff_time: String,
    pub clock_domain_id: String,
    pub clock_basis: ClockBasisV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InformationCutoffSnapshotV1 {
    pub schema: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub cutoff: PredictionInformationCutoffV1,
    pub available_event_ids: Vec<String>,
    pub unavailable_event_ids: Vec<String>,
    pub unaligned_event_ids: Vec<String>,
    pub future_event_ids: Vec<String>,
    pub matured_validated_label_ids: Vec<String>,
    pub not_yet_available_label_ids: Vec<String>,
    pub non_trainable_label_ids: Vec<String>,
    pub unaligned_label_ids: Vec<String>,
    pub claim_limits: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessGraphValidationReportV1 {
    pub schema: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub event_count: usize,
    pub edge_count: usize,
    pub label_count: usize,
    pub validated_label_count: usize,
    pub provisional_label_count: usize,
    pub censored_label_count: usize,
    pub invalid_label_count: usize,
    pub unknown_label_count: usize,
    pub event_ids_are_unique: bool,
    pub edge_endpoints_resolve: bool,
    pub label_measurement_events_resolve: bool,
    pub event_time_is_distinct_from_availability_time: bool,
    pub no_physical_causal_edges_are_inferred: bool,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ProcessEventGraphError(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ProcessEventGraphError> {
    if value.trim().is_empty() {
        Err(ProcessEventGraphError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
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

fn timestamp_micros(
    value: &str,
    basis: &ClockBasisV1,
    field: &str,
) -> Result<i64, ProcessEventGraphError> {
    match basis {
        ClockBasisV1::OffsetAware => {
            let parsed = DateTime::<FixedOffset>::parse_from_rfc3339(value).map_err(|_| {
                ProcessEventGraphError(format!(
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
                    // This is only a comparable integer representation within
                    // the same declared source-local clock domain; not UTC.
                    return Ok(parsed.and_utc().timestamp_micros());
                }
            }
            Err(ProcessEventGraphError(format!(
                "{field}: invalid source-local naive timestamp"
            )))
        }
    }
}

fn optional_identifier(value: Option<&str>, field: &str) -> Result<(), ProcessEventGraphError> {
    if let Some(value) = value {
        require_text(value, field)?;
    }
    Ok(())
}

fn validate_event(event: &ProcessEventV1, index: usize) -> Result<(), ProcessEventGraphError> {
    let prefix = format!("events[{index}]");
    require_text(&event.event_id, &format!("{prefix}.event_id"))?;
    require_text(
        &event.event_clock_domain_id,
        &format!("{prefix}.event_clock_domain_id"),
    )?;
    require_text(
        &event.availability_clock_domain_id,
        &format!("{prefix}.availability_clock_domain_id"),
    )?;
    require_text(&event.authority_ref, &format!("{prefix}.authority_ref"))?;
    if !valid_sha256(&event.source_artifact_sha256) || !valid_sha256(&event.transform_sha256) {
        return Err(ProcessEventGraphError(format!(
            "{prefix}: invalid source or transform SHA-256"
        )));
    }
    for (field, value) in [
        ("subject_id", event.subject_id.as_deref()),
        ("wafer_id", event.wafer_id.as_deref()),
        ("lot_id", event.lot_id.as_deref()),
        ("carrier_id", event.carrier_id.as_deref()),
        ("tool_id", event.tool_id.as_deref()),
        ("chamber_id", event.chamber_id.as_deref()),
        ("module_id", event.module_id.as_deref()),
        ("process_step_id", event.process_step_id.as_deref()),
        ("recipe_id", event.recipe_id.as_deref()),
        ("recipe_revision", event.recipe_revision.as_deref()),
    ] {
        optional_identifier(value, &format!("{prefix}.{field}"))?;
    }
    if event.source_sequence == Some(0) {
        return Err(ProcessEventGraphError(format!(
            "{prefix}.source_sequence: sequence numbers start at one"
        )));
    }
    let event_time = timestamp_micros(
        &event.event_time,
        &event.event_clock_basis,
        &format!("{prefix}.event_time"),
    )?;
    let available_at = timestamp_micros(
        &event.available_at,
        &event.availability_clock_basis,
        &format!("{prefix}.available_at"),
    )?;
    if event.event_clock_domain_id == event.availability_clock_domain_id
        && event.event_clock_basis == event.availability_clock_basis
        && event_time > available_at
    {
        return Err(ProcessEventGraphError(format!(
            "{prefix}: event_time cannot be after available_at in the same clock domain"
        )));
    }
    Ok(())
}

fn validate_label(label: &MetrologyLabelV1, index: usize) -> Result<(), ProcessEventGraphError> {
    let prefix = format!("metrology_labels[{index}]");
    for (field, value) in [
        ("label_id", label.label_id.as_str()),
        ("measurement_event_id", label.measurement_event_id.as_str()),
        ("subject_id", label.subject_id.as_str()),
        ("measurand_id", label.measurand_id.as_str()),
        ("unit", label.unit.as_str()),
        (
            "measurement_clock_domain_id",
            label.measurement_clock_domain_id.as_str(),
        ),
        (
            "availability_clock_domain_id",
            label.availability_clock_domain_id.as_str(),
        ),
        ("authority_ref", label.authority_ref.as_str()),
    ] {
        require_text(value, &format!("{prefix}.{field}"))?;
    }
    if !valid_sha256(&label.source_artifact_sha256) {
        return Err(ProcessEventGraphError(format!(
            "{prefix}.source_artifact_sha256: expected lowercase SHA-256"
        )));
    }
    optional_identifier(
        label.instrument_id.as_deref(),
        &format!("{prefix}.instrument_id"),
    )?;
    optional_identifier(
        label.calibration_ref.as_deref(),
        &format!("{prefix}.calibration_ref"),
    )?;
    if label.revision == 0 {
        return Err(ProcessEventGraphError(format!(
            "{prefix}.revision: revisions start at one"
        )));
    }
    if let Some(supersedes) = label.supersedes_label_id.as_deref() {
        require_text(supersedes, &format!("{prefix}.supersedes_label_id"))?;
        if supersedes == label.label_id {
            return Err(ProcessEventGraphError(format!(
                "{prefix}: a label cannot supersede itself"
            )));
        }
    }

    let measurement_time = timestamp_micros(
        &label.measurement_time,
        &label.measurement_clock_basis,
        &format!("{prefix}.measurement_time"),
    )?;
    let available_at = timestamp_micros(
        &label.available_at,
        &label.availability_clock_basis,
        &format!("{prefix}.available_at"),
    )?;
    if label.measurement_clock_domain_id == label.availability_clock_domain_id
        && label.measurement_clock_basis == label.availability_clock_basis
        && measurement_time > available_at
    {
        return Err(ProcessEventGraphError(format!(
            "{prefix}: measurement_time cannot be after available_at in the same clock domain"
        )));
    }

    if let Some(value) = label.value {
        if !value.is_finite() {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.value: expected finite number"
            )));
        }
    }
    if let Some(uncertainty) = label.standard_uncertainty {
        if !uncertainty.is_finite() || uncertainty < 0.0 {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.standard_uncertainty: expected finite non-negative number"
            )));
        }
    }
    if let Some(bound) = label.censoring_lower_bound {
        if !bound.is_finite() {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.censoring_lower_bound: expected finite number"
            )));
        }
    }
    if let Some(bound) = label.censoring_upper_bound {
        if !bound.is_finite() {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.censoring_upper_bound: expected finite number"
            )));
        }
    }
    if let (Some(lower), Some(upper)) = (label.censoring_lower_bound, label.censoring_upper_bound)
        && lower > upper
    {
        return Err(ProcessEventGraphError(format!(
            "{prefix}: censoring lower bound exceeds upper bound"
        )));
    }

    match &label.status {
        MetrologyLabelStatusV1::Validated => {
            if label.value.is_none() || label.standard_uncertainty.is_none() {
                return Err(ProcessEventGraphError(format!(
                    "{prefix}: validated label requires value and standard uncertainty"
                )));
            }
            if label.censoring_lower_bound.is_some()
                || label.censoring_upper_bound.is_some()
                || label.censoring_basis.is_some()
            {
                return Err(ProcessEventGraphError(format!(
                    "{prefix}: validated point label cannot also be censored"
                )));
            }
        }
        MetrologyLabelStatusV1::Provisional => {
            if label.value.is_none() {
                return Err(ProcessEventGraphError(format!(
                    "{prefix}: provisional label requires a point value"
                )));
            }
        }
        MetrologyLabelStatusV1::Censored => {
            if label.value.is_some()
                || (label.censoring_lower_bound.is_none() && label.censoring_upper_bound.is_none())
            {
                return Err(ProcessEventGraphError(format!(
                    "{prefix}: censored label requires bounds and no point value"
                )));
            }
            require_text(
                label.censoring_basis.as_deref().unwrap_or(""),
                &format!("{prefix}.censoring_basis"),
            )?;
        }
        MetrologyLabelStatusV1::Invalid | MetrologyLabelStatusV1::Unknown => {}
    }
    Ok(())
}

/// Validates graph identities, event/edge/label references and revision lineage.
pub fn validate_process_event_graph_v1(
    graph: &ProcessEventGraphV1,
) -> Result<ProcessGraphValidationReportV1, ProcessEventGraphError> {
    if graph.schema != GRAPH_SCHEMA_V1 {
        return Err(ProcessEventGraphError(
            "schema: unsupported process event graph".into(),
        ));
    }
    require_text(&graph.graph_id, "graph_id")?;
    if !valid_sha256(&graph.source_artifact_sha256) || !valid_sha256(&graph.transform_sha256) {
        return Err(ProcessEventGraphError(
            "graph: invalid source or transform SHA-256".into(),
        ));
    }
    if graph.events.is_empty() {
        return Err(ProcessEventGraphError(
            "events: expected non-empty collection".into(),
        ));
    }

    let mut event_by_id = BTreeMap::new();
    for (index, event) in graph.events.iter().enumerate() {
        validate_event(event, index)?;
        if event_by_id.insert(event.event_id.as_str(), event).is_some() {
            return Err(ProcessEventGraphError(format!(
                "events[{index}].event_id: duplicate"
            )));
        }
    }

    let mut edge_ids = BTreeSet::new();
    for (index, edge) in graph.edges.iter().enumerate() {
        let prefix = format!("edges[{index}]");
        require_text(&edge.edge_id, &format!("{prefix}.edge_id"))?;
        require_text(&edge.source_event_id, &format!("{prefix}.source_event_id"))?;
        require_text(&edge.target_event_id, &format!("{prefix}.target_event_id"))?;
        require_text(&edge.evidence_ref, &format!("{prefix}.evidence_ref"))?;
        if !valid_sha256(&edge.source_artifact_sha256) {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.source_artifact_sha256: invalid SHA-256"
            )));
        }
        if !edge_ids.insert(edge.edge_id.as_str()) {
            return Err(ProcessEventGraphError(format!(
                "{prefix}.edge_id: duplicate"
            )));
        }
        if edge.source_event_id == edge.target_event_id {
            return Err(ProcessEventGraphError(format!(
                "{prefix}: self-edge is not allowed"
            )));
        }
        if !event_by_id.contains_key(edge.source_event_id.as_str())
            || !event_by_id.contains_key(edge.target_event_id.as_str())
        {
            return Err(ProcessEventGraphError(format!(
                "{prefix}: edge endpoint does not resolve to an event"
            )));
        }
        if matches!(&edge.edge_kind, ProcessEdgeKindV1::Precedes) {
            let source = event_by_id[edge.source_event_id.as_str()];
            let target = event_by_id[edge.target_event_id.as_str()];
            if source.event_clock_domain_id == target.event_clock_domain_id
                && source.event_clock_basis == target.event_clock_basis
            {
                let source_time = timestamp_micros(
                    &source.event_time,
                    &source.event_clock_basis,
                    "edge source time",
                )?;
                let target_time = timestamp_micros(
                    &target.event_time,
                    &target.event_clock_basis,
                    "edge target time",
                )?;
                if source_time > target_time {
                    return Err(ProcessEventGraphError(format!(
                        "{prefix}: Precedes edge contradicts source-clock event time"
                    )));
                }
            }
        }
    }

    let mut label_by_id = BTreeMap::new();
    let mut counts = [0usize; 5];
    for (index, label) in graph.metrology_labels.iter().enumerate() {
        validate_label(label, index)?;
        if label_by_id.insert(label.label_id.as_str(), label).is_some() {
            return Err(ProcessEventGraphError(format!(
                "metrology_labels[{index}].label_id: duplicate"
            )));
        }
        let measurement_event = event_by_id
            .get(label.measurement_event_id.as_str())
            .ok_or_else(|| {
                ProcessEventGraphError(format!(
                    "metrology_labels[{index}].measurement_event_id: unresolved event"
                ))
            })?;
        if measurement_event.event_kind != ProcessEventKindV1::Metrology {
            return Err(ProcessEventGraphError(format!(
                "metrology_labels[{index}]: measurement event is not a metrology event"
            )));
        }
        if measurement_event.subject_id.as_deref() != Some(label.subject_id.as_str()) {
            return Err(ProcessEventGraphError(format!(
                "metrology_labels[{index}]: subject does not match measurement event"
            )));
        }
        match label.status {
            MetrologyLabelStatusV1::Validated => counts[0] += 1,
            MetrologyLabelStatusV1::Provisional => counts[1] += 1,
            MetrologyLabelStatusV1::Censored => counts[2] += 1,
            MetrologyLabelStatusV1::Invalid => counts[3] += 1,
            MetrologyLabelStatusV1::Unknown => counts[4] += 1,
        }
    }

    for (index, label) in graph.metrology_labels.iter().enumerate() {
        if let Some(parent_id) = label.supersedes_label_id.as_deref() {
            let parent = label_by_id.get(parent_id).ok_or_else(|| {
                ProcessEventGraphError(format!(
                    "metrology_labels[{index}].supersedes_label_id: unknown parent {parent_id}"
                ))
            })?;
            if parent.subject_id != label.subject_id
                || parent.measurand_id != label.measurand_id
                || parent.revision >= label.revision
            {
                return Err(ProcessEventGraphError(format!(
                    "metrology_labels[{index}]: superseding revision must retain subject/measurand and increase revision"
                )));
            }
        }
    }

    let serialized = serde_json::to_vec(graph)
        .map_err(|error| ProcessEventGraphError(format!("graph serialization failed: {error}")))?;
    Ok(ProcessGraphValidationReportV1 {
        schema: "luminous.semi-bench.process-event-graph-validation.v1".into(),
        graph_id: graph.graph_id.clone(),
        graph_sha256: sha256_hex(&serialized),
        event_count: graph.events.len(),
        edge_count: graph.edges.len(),
        label_count: graph.metrology_labels.len(),
        validated_label_count: counts[0],
        provisional_label_count: counts[1],
        censored_label_count: counts[2],
        invalid_label_count: counts[3],
        unknown_label_count: counts[4],
        event_ids_are_unique: true,
        edge_endpoints_resolve: true,
        label_measurement_events_resolve: true,
        event_time_is_distinct_from_availability_time: true,
        no_physical_causal_edges_are_inferred: true,
    })
}

/// Creates an information snapshot at a real prediction/training cutoff.
///
/// Events and labels from unaligned clock domains are explicitly listed as
/// unaligned and excluded. Labels are eligible only if they are VALIDATED and
/// available by the cutoff; measurement time alone never grants eligibility.
pub fn build_information_cutoff_snapshot_v1(
    graph: &ProcessEventGraphV1,
    cutoff: &PredictionInformationCutoffV1,
) -> Result<InformationCutoffSnapshotV1, ProcessEventGraphError> {
    let validation = validate_process_event_graph_v1(graph)?;
    require_text(&cutoff.cutoff_id, "cutoff.cutoff_id")?;
    require_text(&cutoff.clock_domain_id, "cutoff.clock_domain_id")?;
    let cutoff_micros = timestamp_micros(
        &cutoff.cutoff_time,
        &cutoff.clock_basis,
        "cutoff.cutoff_time",
    )?;

    let mut available_event_ids = Vec::new();
    let mut unavailable_event_ids = Vec::new();
    let mut unaligned_event_ids = Vec::new();
    let mut future_event_ids = Vec::new();
    for event in &graph.events {
        if event.availability_clock_domain_id != cutoff.clock_domain_id
            || event.availability_clock_basis != cutoff.clock_basis
            || event.event_clock_domain_id != cutoff.clock_domain_id
            || event.event_clock_basis != cutoff.clock_basis
        {
            unaligned_event_ids.push(event.event_id.clone());
            continue;
        }
        let available_at = timestamp_micros(
            &event.available_at,
            &event.availability_clock_basis,
            "event.available_at",
        )?;
        if available_at > cutoff_micros {
            unavailable_event_ids.push(event.event_id.clone());
            continue;
        }
        let event_time = timestamp_micros(
            &event.event_time,
            &event.event_clock_basis,
            "event.event_time",
        )?;
        if event_time > cutoff_micros {
            future_event_ids.push(event.event_id.clone());
            continue;
        }
        available_event_ids.push(event.event_id.clone());
    }

    let mut matured_validated_label_ids = Vec::new();
    let mut not_yet_available_label_ids = Vec::new();
    let mut non_trainable_label_ids = Vec::new();
    let mut unaligned_label_ids = Vec::new();
    for label in &graph.metrology_labels {
        if label.availability_clock_domain_id != cutoff.clock_domain_id
            || label.availability_clock_basis != cutoff.clock_basis
            || label.measurement_clock_domain_id != cutoff.clock_domain_id
            || label.measurement_clock_basis != cutoff.clock_basis
        {
            unaligned_label_ids.push(label.label_id.clone());
            continue;
        }
        let available_at = timestamp_micros(
            &label.available_at,
            &label.availability_clock_basis,
            "label.available_at",
        )?;
        if available_at > cutoff_micros {
            not_yet_available_label_ids.push(label.label_id.clone());
        } else if matches!(&label.status, MetrologyLabelStatusV1::Validated) {
            matured_validated_label_ids.push(label.label_id.clone());
        } else {
            non_trainable_label_ids.push(label.label_id.clone());
        }
    }

    available_event_ids.sort();
    unavailable_event_ids.sort();
    unaligned_event_ids.sort();
    future_event_ids.sort();
    matured_validated_label_ids.sort();
    not_yet_available_label_ids.sort();
    non_trainable_label_ids.sort();
    unaligned_label_ids.sort();

    Ok(InformationCutoffSnapshotV1 {
        schema: SNAPSHOT_SCHEMA_V1.into(),
        graph_id: graph.graph_id.clone(),
        graph_sha256: validation.graph_sha256,
        cutoff: cutoff.clone(),
        available_event_ids,
        unavailable_event_ids,
        unaligned_event_ids,
        future_event_ids,
        matured_validated_label_ids,
        not_yet_available_label_ids,
        non_trainable_label_ids,
        unaligned_label_ids,
        claim_limits: vec![
            "Only event records available in the declared cutoff clock domain are included".into(),
            "Metrology labels are trainable only after available_at and only in validated state"
                .into(),
            "Event-time and availability-time are distinct and not interchangeable".into(),
            "Clock-domain mismatches are excluded rather than coerced".into(),
            "Process sequence edges are evidence-backed ordering/context, not causal discovery"
                .into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn event(
        id: &str,
        kind: ProcessEventKindV1,
        event_time: &str,
        available_at: &str,
    ) -> ProcessEventV1 {
        ProcessEventV1 {
            event_id: id.into(),
            event_kind: kind,
            subject_id: Some("wafer-17".into()),
            wafer_id: Some("wafer-17".into()),
            lot_id: Some("lot-4".into()),
            carrier_id: None,
            tool_id: Some("etch-7".into()),
            chamber_id: Some("chamber-a".into()),
            module_id: Some("module-1".into()),
            process_step_id: Some("etch-main".into()),
            recipe_id: Some("recipe-x".into()),
            recipe_revision: Some("r12".into()),
            event_time: event_time.into(),
            event_clock_domain_id: "fab-local".into(),
            event_clock_basis: ClockBasisV1::SourceLocalNaive,
            available_at: available_at.into(),
            availability_clock_domain_id: "fab-local".into(),
            availability_clock_basis: ClockBasisV1::SourceLocalNaive,
            source_sequence: Some(1),
            source_artifact_sha256: SHA_A.into(),
            transform_sha256: SHA_B.into(),
            authority_ref: "source-eda-adapter-v1".into(),
        }
    }

    fn label(
        id: &str,
        status: MetrologyLabelStatusV1,
        available_at: &str,
        revision: u64,
        supersedes: Option<&str>,
    ) -> MetrologyLabelV1 {
        let validated = matches!(&status, MetrologyLabelStatusV1::Validated);
        MetrologyLabelV1 {
            label_id: id.into(),
            measurement_event_id: "met-17".into(),
            subject_id: "wafer-17".into(),
            measurand_id: "film-thickness".into(),
            value: if validated { Some(12.5) } else { None },
            unit: "nm".into(),
            standard_uncertainty: if validated { Some(0.2) } else { None },
            censoring_lower_bound: None,
            censoring_upper_bound: None,
            censoring_basis: None,
            status,
            measurement_time: "2026-11-22T09:00:00".into(),
            measurement_clock_domain_id: "fab-local".into(),
            measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
            available_at: available_at.into(),
            availability_clock_domain_id: "fab-local".into(),
            availability_clock_basis: ClockBasisV1::SourceLocalNaive,
            instrument_id: Some("ellipsometer-2".into()),
            calibration_ref: Some("cal-2026-11".into()),
            revision,
            supersedes_label_id: supersedes.map(str::to_string),
            source_artifact_sha256: SHA_C.into(),
            authority_ref: "metrology-lims-validated".into(),
        }
    }

    fn graph() -> ProcessEventGraphV1 {
        ProcessEventGraphV1 {
            schema: GRAPH_SCHEMA_V1.into(),
            graph_id: "graph-17".into(),
            source_artifact_sha256: SHA_A.into(),
            transform_sha256: SHA_B.into(),
            events: vec![
                event(
                    "process-17",
                    ProcessEventKindV1::ProcessStepEnd,
                    "2026-11-22T08:00:00",
                    "2026-11-22T08:00:01",
                ),
                event(
                    "met-17",
                    ProcessEventKindV1::Metrology,
                    "2026-11-22T09:00:00",
                    "2026-11-22T09:00:02",
                ),
            ],
            edges: vec![ProcessEventEdgeV1 {
                edge_id: "edge-1".into(),
                source_event_id: "process-17".into(),
                target_event_id: "met-17".into(),
                edge_kind: ProcessEdgeKindV1::Precedes,
                evidence_ref: "wafer-history-record-1".into(),
                source_artifact_sha256: SHA_C.into(),
            }],
            metrology_labels: vec![
                label(
                    "label-validated",
                    MetrologyLabelStatusV1::Validated,
                    "2026-11-22T10:00:00",
                    1,
                    None,
                ),
                label(
                    "label-delayed",
                    MetrologyLabelStatusV1::Validated,
                    "2026-11-22T12:00:00",
                    1,
                    None,
                ),
                label(
                    "label-provisional",
                    MetrologyLabelStatusV1::Provisional,
                    "2026-11-22T09:30:00",
                    1,
                    None,
                ),
            ],
        }
    }

    #[test]
    fn graph_validation_checks_edges_labels_and_revision_lineage() {
        let report = validate_process_event_graph_v1(&graph()).unwrap();
        assert_eq!(report.event_count, 2);
        assert_eq!(report.edge_count, 1);
        assert_eq!(report.label_count, 3);
        assert_eq!(report.validated_label_count, 2);
        assert_eq!(report.provisional_label_count, 1);
        assert!(report.edge_endpoints_resolve);
        assert!(report.label_measurement_events_resolve);
        assert!(report.no_physical_causal_edges_are_inferred);
    }

    #[test]
    fn label_is_not_trainable_before_result_availability_time() {
        let input = graph();
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "train-at-11".into(),
            cutoff_time: "2026-11-22T11:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let snapshot = build_information_cutoff_snapshot_v1(&input, &cutoff).unwrap();
        assert_eq!(snapshot.matured_validated_label_ids, ["label-validated"]);
        assert_eq!(snapshot.not_yet_available_label_ids, ["label-delayed"]);
        assert_eq!(snapshot.non_trainable_label_ids, ["label-provisional"]);
    }

    #[test]
    fn late_arriving_event_is_excluded_even_if_physical_event_time_is_old() {
        let mut input = graph();
        input.events.push(event(
            "late-old-event",
            ProcessEventKindV1::Alarm,
            "2026-11-22T07:00:00",
            "2026-11-22T13:00:00",
        ));
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "predict-at-11".into(),
            cutoff_time: "2026-11-22T11:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let snapshot = build_information_cutoff_snapshot_v1(&input, &cutoff).unwrap();
        assert!(
            snapshot
                .unavailable_event_ids
                .contains(&"late-old-event".into())
        );
        assert!(
            !snapshot
                .available_event_ids
                .contains(&"late-old-event".into())
        );
    }

    #[test]
    fn mismatched_clock_domains_are_explicitly_excluded() {
        let mut input = graph();
        input.events[0].availability_clock_domain_id = "collector-clock".into();
        input.metrology_labels[0].availability_clock_domain_id = "lims-clock".into();
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "cutoff-1".into(),
            cutoff_time: "2026-11-22T11:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let snapshot = build_information_cutoff_snapshot_v1(&input, &cutoff).unwrap();
        assert!(snapshot.unaligned_event_ids.contains(&"process-17".into()));
        assert!(
            snapshot
                .unaligned_label_ids
                .contains(&"label-validated".into())
        );
    }

    #[test]
    fn event_time_and_measurement_time_clocks_must_align_with_cutoff() {
        let mut input = graph();
        input.events[0].event_clock_domain_id = "tool-clock".into();
        input.metrology_labels[0].measurement_clock_domain_id = "lims-instrument-clock".into();
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "cutoff-aligned-fab".into(),
            cutoff_time: "2026-11-22T11:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let snapshot = build_information_cutoff_snapshot_v1(&input, &cutoff).unwrap();
        assert!(snapshot.unaligned_event_ids.contains(&"process-17".into()));
        assert!(
            snapshot
                .unaligned_label_ids
                .contains(&"label-validated".into())
        );
        assert!(!snapshot.available_event_ids.contains(&"process-17".into()));
        assert!(
            !snapshot
                .matured_validated_label_ids
                .contains(&"label-validated".into())
        );
    }

    #[test]
    fn event_cannot_be_visible_before_it_occurs_in_the_same_clock_domain() {
        let mut input = graph();
        input.events[0].available_at = "2026-11-22T07:59:59".into();
        assert!(validate_process_event_graph_v1(&input).is_err());
    }

    #[test]
    fn edge_endpoints_and_label_subjects_are_checked() {
        let mut invalid_edge = graph();
        invalid_edge.edges[0].target_event_id = "missing-event".into();
        assert!(validate_process_event_graph_v1(&invalid_edge).is_err());

        let mut invalid_label = graph();
        invalid_label.metrology_labels[0].subject_id = "wafer-wrong".into();
        assert!(validate_process_event_graph_v1(&invalid_label).is_err());
    }

    #[test]
    fn superseding_label_must_retain_target_and_increase_revision() {
        let mut input = graph();
        let mut revision = label(
            "label-revised",
            MetrologyLabelStatusV1::Validated,
            "2026-11-22T11:00:00",
            2,
            Some("label-validated"),
        );
        revision.value = Some(12.7);
        input.metrology_labels.push(revision);
        assert!(validate_process_event_graph_v1(&input).is_ok());

        let mut invalid = graph();
        let mut bad_revision = label(
            "label-revised",
            MetrologyLabelStatusV1::Validated,
            "2026-11-22T11:00:00",
            1,
            Some("label-validated"),
        );
        bad_revision.subject_id = "wafer-99".into();
        invalid.metrology_labels.push(bad_revision);
        assert!(validate_process_event_graph_v1(&invalid).is_err());
    }

    #[test]
    fn offset_aware_timestamp_requires_explicit_timezone() {
        let mut input = graph();
        input.events[0].event_clock_basis = ClockBasisV1::OffsetAware;
        input.events[0].event_time = "2026-11-22T08:00:00".into();
        assert!(validate_process_event_graph_v1(&input).is_err());
    }
}
