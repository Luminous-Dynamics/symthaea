// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-bound, multi-axis MUSE benchmark reports.
//!
//! The validator does not compute audio/score metrics. It ensures that results
//! are accompanied by immutable dataset, reference, split, track, metric and
//! renderer identities, and that comparisons are only made across identical
//! protocols. It deliberately has no aggregate "overall quality" score.

use crate::evidence_digest::{canonical_json_sha256, decode_hex_32};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MULTI_AXIS_BENCHMARK_SCHEMA_V1: &str = "muse-multi-axis-benchmark-v1";
pub const MULTI_AXIS_PROTOCOL_SCHEMA_V1: &str = "muse-benchmark-protocol-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationTrackV1 {
    Blind,
    StavesInformed,
    HumanAssisted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricAxisV1 {
    NotationFidelity,
    StructureFidelity,
    PlaybackFidelity,
    BeatPerception,
    Generalization,
    HumanUtility,
    Uncertainty,
    Efficiency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirectionV1 {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricUnitV1 {
    Fraction,
    Percent,
    Count,
    Milliseconds,
    Seconds,
    Ratio,
    Bytes,
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupingUnitV1 {
    Work,
    Composer,
    Artist,
    Performer,
    RecordingSession,
    InstrumentFamily,
    Genre,
    Culture,
    RandomExcerpt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneralizationClaimV1 {
    InDistribution,
    OutOfDistribution,
}

/// Explicit split provenance. Group IDs are opaque dataset-native identifiers,
/// not user-facing names. They are committed by the protocol fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupSplitV1 {
    pub claim: GeneralizationClaimV1,
    pub grouping_unit: GroupingUnitV1,
    pub split_manifest_sha256: String,
    pub training_group_ids: Vec<String>,
    pub evaluation_group_ids: Vec<String>,
    pub held_out_group_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricToleranceV1 {
    pub value: f64,
    pub unit: String,
    pub matching_rule: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricDefinitionV1 {
    pub metric_id: String,
    pub axis: MetricAxisV1,
    pub direction: MetricDirectionV1,
    pub unit: MetricUnitV1,
    pub protocol_version: String,
    pub tolerance: Option<MetricToleranceV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnotationProtocolV1 {
    pub protocol_id: String,
    pub version: String,
    pub reference_manifest_sha256: String,
    pub reference_count: u32,
    pub annotator_description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererIdentityV1 {
    pub renderer_id: String,
    pub renderer_version: String,
    pub soundfont_or_asset_sha256: Option<String>,
    pub render_configuration_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_layout: String,
    pub loudness_policy: String,
    pub tempo_map_policy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkProtocolV1 {
    pub schema: String,
    pub protocol_id: String,
    pub benchmark_family: String,
    pub dataset_name: String,
    pub dataset_version: String,
    pub dataset_manifest_sha256: String,
    pub training_corpus_manifest_sha256: String,
    pub evaluation_track: EvaluationTrackV1,
    pub split: GroupSplitV1,
    pub annotation: AnnotationProtocolV1,
    pub renderer: Option<RendererIdentityV1>,
    pub metric_definitions: Vec<MetricDefinitionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemIdentityV1 {
    pub system_id: String,
    pub source_commit: String,
    pub build_sha256: String,
    pub model_sha256: Option<String>,
    pub model_parameter_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusteredConfidenceIntervalV1 {
    pub lower: f64,
    pub upper: f64,
    /// For example 0.95 for a 95% interval.
    pub confidence_level: f64,
    pub cluster_unit: GroupingUnitV1,
    pub cluster_count: u32,
    pub method: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricValueV1 {
    pub metric_id: String,
    pub value: f64,
    /// Number of samples included in this metric value.
    pub sample_count: u64,
    /// Digest of the exact sample IDs included, preventing silent cherry-picking.
    pub evaluated_sample_manifest_sha256: String,
    pub confidence_interval: Option<ClusteredConfidenceIntervalV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSliceV1 {
    pub slice_id: String,
    pub is_overall: bool,
    pub grouping_unit: Option<GroupingUnitV1>,
    pub grouping_value: Option<String>,
    pub sample_count: u64,
    pub metrics: Vec<MetricValueV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceReceiptV1 {
    pub inference_milliseconds: Option<f64>,
    pub real_time_factor: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
    pub energy_watt_hours: Option<f64>,
    pub hardware_description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRunReportV1 {
    pub schema: String,
    pub protocol: BenchmarkProtocolV1,
    pub protocol_fingerprint: String,
    pub system: SystemIdentityV1,
    pub source_artifact_sha256: String,
    pub model_input_manifest_sha256: String,
    pub slices: Vec<BenchmarkSliceV1>,
    pub resources: ResourceReceiptV1,
    /// Canonical digest of this report with this field cleared.
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkValidationError {
    InvalidSchema,
    EmptyIdentity { field: String },
    InvalidDigest { field: String },
    DuplicateMetricId { metric_id: String },
    DuplicateGroupId { group_id: String },
    TrainingEvaluationOverlap { group_id: String },
    MissingHeldOutGroups,
    InvalidSplitPolicy,
    HeldOutGroupNotEvaluated { group_id: String },
    RandomExcerptCannotClaimOutOfDistribution,
    PlaybackMetricRequiresRenderer,
    InvalidMetricTolerance { metric_id: String },
    UnknownMetric { metric_id: String },
    DuplicateMetricValue { slice_id: String, metric_id: String },
    MissingOverallSlice,
    DuplicateOverallSlice,
    InvalidMetricValue { metric_id: String },
    InvalidConfidenceInterval { metric_id: String },
    InvalidSlice { slice_id: String },
    InvalidResourceReceipt { field: String },
    ProtocolFingerprintMismatch,
    ReportDigestMismatch,
    IncompatibleProtocol,
    MissingComparableMetric { metric_id: String },
    MissingMetricCoverage { metric_id: String },
    MissingGroupSlice { group_id: String },
    MissingGroupMetricCoverage { group_id: String, metric_id: String },
    DuplicateGroupSlice { group_id: String },
    IncompatibleMetricCoverage { metric_id: String },
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkComparisonV1 {
    pub protocol_fingerprint: String,
    pub metric_id: String,
    pub unit: MetricUnitV1,
    pub direction: MetricDirectionV1,
    pub left_system_id: String,
    pub right_system_id: String,
    pub left_value: f64,
    pub right_value: f64,
    pub winner_system_id: Option<String>,
}

/// Canonical fingerprint of the full protocol, including dataset, split,
/// references, renderer, evaluation track, metrics and tolerances.
pub fn benchmark_protocol_fingerprint(
    protocol: &BenchmarkProtocolV1,
) -> Result<String, BenchmarkValidationError> {
    validate_benchmark_protocol(protocol)?;
    canonical_json_sha256(protocol).map_err(|_| BenchmarkValidationError::SerializationFailed)
}

pub fn validate_benchmark_protocol(
    protocol: &BenchmarkProtocolV1,
) -> Result<(), BenchmarkValidationError> {
    if protocol.schema != MULTI_AXIS_PROTOCOL_SCHEMA_V1 {
        return Err(BenchmarkValidationError::InvalidSchema);
    }
    for (field, value) in [
        ("protocol_id", protocol.protocol_id.as_str()),
        ("benchmark_family", protocol.benchmark_family.as_str()),
        ("dataset_name", protocol.dataset_name.as_str()),
        ("dataset_version", protocol.dataset_version.as_str()),
        ("annotation_protocol_id", protocol.annotation.protocol_id.as_str()),
        ("annotation_version", protocol.annotation.version.as_str()),
        ("annotator_description", protocol.annotation.annotator_description.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(BenchmarkValidationError::EmptyIdentity {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("dataset_manifest_sha256", protocol.dataset_manifest_sha256.as_str()),
        (
            "training_corpus_manifest_sha256",
            protocol.training_corpus_manifest_sha256.as_str(),
        ),
        (
            "split_manifest_sha256",
            protocol.split.split_manifest_sha256.as_str(),
        ),
        (
            "reference_manifest_sha256",
            protocol.annotation.reference_manifest_sha256.as_str(),
        ),
    ] {
        validate_digest(field, digest)?;
    }
    if protocol.annotation.reference_count == 0 {
        return Err(BenchmarkValidationError::EmptyIdentity {
            field: "annotation.reference_count".into(),
        });
    }

    validate_group_ids(
        &protocol.split.training_group_ids,
        "training_group_ids",
    )?;
    validate_group_ids(
        &protocol.split.evaluation_group_ids,
        "evaluation_group_ids",
    )?;
    validate_group_ids(
        &protocol.split.held_out_group_ids,
        "held_out_group_ids",
    )?;
    let training: BTreeSet<String> =
        protocol.split.training_group_ids.iter().cloned().collect();
    let evaluation: BTreeSet<String> =
        protocol.split.evaluation_group_ids.iter().cloned().collect();
    if let Some(group_id) = training.intersection(&evaluation).next() {
        return Err(BenchmarkValidationError::TrainingEvaluationOverlap {
            group_id: group_id.clone(),
        });
    }
    if training.is_empty() || evaluation.is_empty() {
        return Err(BenchmarkValidationError::InvalidSplitPolicy);
    }
    match protocol.split.claim {
        GeneralizationClaimV1::InDistribution => {
            // In-distribution runs may still hold out works or recording
            // sessions, provided the broader composer/performer distribution
            // is not being claimed as unseen. The grouping unit is explicit.
            if !matches!(
                protocol.split.grouping_unit,
                GroupingUnitV1::RandomExcerpt
                    | GroupingUnitV1::Work
                    | GroupingUnitV1::RecordingSession
            ) || !protocol.split.held_out_group_ids.is_empty()
            {
                return Err(BenchmarkValidationError::InvalidSplitPolicy);
            }
        }
        GeneralizationClaimV1::OutOfDistribution => {
            if protocol.split.grouping_unit == GroupingUnitV1::RandomExcerpt {
                return Err(BenchmarkValidationError::RandomExcerptCannotClaimOutOfDistribution);
            }
            if protocol.split.held_out_group_ids.is_empty() {
                return Err(BenchmarkValidationError::MissingHeldOutGroups);
            }
            let held_out: BTreeSet<String> =
                protocol.split.held_out_group_ids.iter().cloned().collect();
            if held_out != evaluation {
                return Err(BenchmarkValidationError::InvalidSplitPolicy);
            }
            for group_id in &protocol.split.held_out_group_ids {
                if !evaluation.contains(group_id) || training.contains(group_id) {
                    return Err(BenchmarkValidationError::HeldOutGroupNotEvaluated {
                        group_id: group_id.clone(),
                    });
                }
            }
        }
    }

    let mut metric_ids = BTreeSet::new();
    let mut needs_renderer = false;
    if protocol.metric_definitions.is_empty() {
        return Err(BenchmarkValidationError::EmptyIdentity {
            field: "metric_definitions".into(),
        });
    }
    for definition in &protocol.metric_definitions {
        if definition.metric_id.trim().is_empty() || definition.protocol_version.trim().is_empty() {
            return Err(BenchmarkValidationError::EmptyIdentity {
                field: "metric_definitions.metric_id_or_protocol_version".into(),
            });
        }
        if !metric_ids.insert(definition.metric_id.as_str()) {
            return Err(BenchmarkValidationError::DuplicateMetricId {
                metric_id: definition.metric_id.clone(),
            });
        }
        if definition.axis == MetricAxisV1::PlaybackFidelity {
            needs_renderer = true;
        }
        if let Some(tolerance) = &definition.tolerance {
            if !tolerance.value.is_finite()
                || tolerance.value < 0.0
                || tolerance.unit.trim().is_empty()
                || tolerance.matching_rule.trim().is_empty()
            {
                return Err(BenchmarkValidationError::InvalidMetricTolerance {
                    metric_id: definition.metric_id.clone(),
                });
            }
        }
    }
    if needs_renderer && protocol.renderer.is_none() {
        return Err(BenchmarkValidationError::PlaybackMetricRequiresRenderer);
    }
    if let Some(renderer) = &protocol.renderer {
        for (field, value) in [
            ("renderer_id", renderer.renderer_id.as_str()),
            ("renderer_version", renderer.renderer_version.as_str()),
            ("channel_layout", renderer.channel_layout.as_str()),
            ("loudness_policy", renderer.loudness_policy.as_str()),
            ("tempo_map_policy", renderer.tempo_map_policy.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BenchmarkValidationError::EmptyIdentity {
                    field: field.into(),
                });
            }
        }
        validate_digest(
            "renderer.render_configuration_sha256",
            &renderer.render_configuration_sha256,
        )?;
        if renderer.sample_rate_hz == 0 {
            return Err(BenchmarkValidationError::EmptyIdentity {
                field: "renderer.sample_rate_hz".into(),
            });
        }
        if let Some(digest) = renderer.soundfont_or_asset_sha256.as_deref() {
            validate_digest("renderer.soundfont_or_asset_sha256", digest)?;
        }
    }
    Ok(())
}

fn validate_group_ids(
    group_ids: &[String],
    field: &str,
) -> Result<(), BenchmarkValidationError> {
    let mut seen = BTreeSet::new();
    for group_id in group_ids {
        if group_id.trim().is_empty() {
            return Err(BenchmarkValidationError::EmptyIdentity {
                field: field.into(),
            });
        }
        if !seen.insert(group_id.as_str()) {
            return Err(BenchmarkValidationError::DuplicateGroupId {
                group_id: group_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_digest(field: &str, digest: &str) -> Result<(), BenchmarkValidationError> {
    if decode_hex_32(digest).is_none() {
        return Err(BenchmarkValidationError::InvalidDigest {
            field: field.into(),
        });
    }
    Ok(())
}

pub fn benchmark_run_report_commitment(
    report: &BenchmarkRunReportV1,
) -> Result<String, BenchmarkValidationError> {
    let mut payload = report.clone();
    payload.report_sha256.clear();
    canonical_json_sha256(&payload).map_err(|_| BenchmarkValidationError::SerializationFailed)
}

pub fn validate_benchmark_run_report(
    report: &BenchmarkRunReportV1,
) -> Result<(), BenchmarkValidationError> {
    if report.schema != MULTI_AXIS_BENCHMARK_SCHEMA_V1 {
        return Err(BenchmarkValidationError::InvalidSchema);
    }
    validate_benchmark_protocol(&report.protocol)?;
    let expected_protocol_fingerprint = benchmark_protocol_fingerprint(&report.protocol)?;
    if expected_protocol_fingerprint != report.protocol_fingerprint {
        return Err(BenchmarkValidationError::ProtocolFingerprintMismatch);
    }
    for (field, value) in [
        ("system_id", report.system.system_id.as_str()),
        ("source_commit", report.system.source_commit.as_str()),
        ("hardware_description", report.resources.hardware_description.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(BenchmarkValidationError::EmptyIdentity {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("system.build_sha256", report.system.build_sha256.as_str()),
        ("source_artifact_sha256", report.source_artifact_sha256.as_str()),
        (
            "model_input_manifest_sha256",
            report.model_input_manifest_sha256.as_str(),
        ),
    ] {
        validate_digest(field, digest)?;
    }
    if let Some(digest) = report.system.model_sha256.as_deref() {
        validate_digest("system.model_sha256", digest)?;
    }
    let definitions: std::collections::BTreeMap<_, _> = report
        .protocol
        .metric_definitions
        .iter()
        .map(|definition| (definition.metric_id.as_str(), definition))
        .collect();
    let mut overall_count = 0usize;
    let mut overall_metric_ids = BTreeSet::new();
    let mut group_slice_keys = BTreeSet::new();
    let mut slice_ids = BTreeSet::new();
    for slice in &report.slices {
        if slice.slice_id.trim().is_empty() || slice.sample_count == 0 {
            return Err(BenchmarkValidationError::InvalidSlice {
                slice_id: slice.slice_id.clone(),
            });
        }
        if !slice_ids.insert(slice.slice_id.as_str()) {
            return Err(BenchmarkValidationError::InvalidSlice {
                slice_id: slice.slice_id.clone(),
            });
        }
        if slice.is_overall {
            overall_count += 1;
            if slice.grouping_unit.is_some() || slice.grouping_value.is_some() {
                return Err(BenchmarkValidationError::InvalidSlice {
                    slice_id: slice.slice_id.clone(),
                });
            }
        } else if slice.grouping_unit.is_none()
            || slice.grouping_value.as_deref().is_none_or(str::is_empty)
        {
            return Err(BenchmarkValidationError::InvalidSlice {
                slice_id: slice.slice_id.clone(),
            });
        } else {
            let key = (
                slice.grouping_unit.unwrap(),
                slice.grouping_value.as_deref().unwrap(),
            );
            if !group_slice_keys.insert(key) {
                return Err(BenchmarkValidationError::DuplicateGroupSlice {
                    group_id: key.1.to_string(),
                });
            }
        }
        let mut metric_ids = BTreeSet::new();
        for metric in &slice.metrics {
            let Some(definition) = definitions.get(metric.metric_id.as_str()) else {
                return Err(BenchmarkValidationError::UnknownMetric {
                    metric_id: metric.metric_id.clone(),
                });
            };
            if !metric_ids.insert(metric.metric_id.as_str()) {
                return Err(BenchmarkValidationError::DuplicateMetricValue {
                    slice_id: slice.slice_id.clone(),
                    metric_id: metric.metric_id.clone(),
                });
            }
            if slice.is_overall {
                overall_metric_ids.insert(metric.metric_id.as_str());
            }
            validate_digest(
                "metric.evaluated_sample_manifest_sha256",
                &metric.evaluated_sample_manifest_sha256,
            )?;
            if !metric.value.is_finite()
                || metric.sample_count == 0
                || metric.sample_count > slice.sample_count
            {
                return Err(BenchmarkValidationError::InvalidMetricValue {
                    metric_id: metric.metric_id.clone(),
                });
            }
            if definition.unit == MetricUnitV1::Fraction
                && !(0.0..=1.0).contains(&metric.value)
            {
                return Err(BenchmarkValidationError::InvalidMetricValue {
                    metric_id: metric.metric_id.clone(),
                });
            }
            if slice.is_overall
                && report.protocol.split.claim == GeneralizationClaimV1::OutOfDistribution
                && metric.confidence_interval.is_none()
            {
                return Err(BenchmarkValidationError::InvalidConfidenceInterval {
                    metric_id: metric.metric_id.clone(),
                });
            }
            if let Some(interval) = &metric.confidence_interval {
                if (slice.is_overall
                    && report.protocol.split.claim == GeneralizationClaimV1::OutOfDistribution
                    && interval.cluster_unit != report.protocol.split.grouping_unit)
                    || !interval.lower.is_finite()
                    || !interval.upper.is_finite()
                    || !interval.confidence_level.is_finite()
                    || interval.lower > interval.upper
                    || metric.value < interval.lower
                    || metric.value > interval.upper
                    || !(0.0..=1.0).contains(&interval.confidence_level)
                    || interval.cluster_count < 2
                    || interval.method.trim().is_empty()
                {
                    return Err(BenchmarkValidationError::InvalidConfidenceInterval {
                        metric_id: metric.metric_id.clone(),
                    });
                }
            }
        }
    }
    match overall_count {
        0 => return Err(BenchmarkValidationError::MissingOverallSlice),
        1 => {}
        _ => return Err(BenchmarkValidationError::DuplicateOverallSlice),
    }
    let expected_metric_ids: BTreeSet<_> = definitions.keys().copied().collect();
    if overall_metric_ids != expected_metric_ids {
        if let Some(metric_id) = expected_metric_ids.difference(&overall_metric_ids).next() {
            return Err(BenchmarkValidationError::MissingMetricCoverage {
                metric_id: (*metric_id).to_string(),
            });
        }
        return Err(BenchmarkValidationError::InvalidSlice {
            slice_id: "overall".into(),
        });
    }
    if report.protocol.split.claim == GeneralizationClaimV1::OutOfDistribution {
        for group_id in &report.protocol.split.held_out_group_ids {
            let Some(group_slice) = report.slices.iter().find(|slice| {
                !slice.is_overall
                    && slice.grouping_unit == Some(report.protocol.split.grouping_unit)
                    && slice.grouping_value.as_deref() == Some(group_id.as_str())
            }) else {
                return Err(BenchmarkValidationError::MissingGroupSlice {
                    group_id: group_id.clone(),
                });
            };
            let group_metric_ids: BTreeSet<_> = group_slice
                .metrics
                .iter()
                .map(|metric| metric.metric_id.as_str())
                .collect();
            for metric_id in &expected_metric_ids {
                if !group_metric_ids.contains(metric_id) {
                    return Err(BenchmarkValidationError::MissingGroupMetricCoverage {
                        group_id: group_id.clone(),
                        metric_id: (*metric_id).to_string(),
                    });
                }
            }
        }
    }
    for (field, value) in [
        ("inference_milliseconds", report.resources.inference_milliseconds),
        ("real_time_factor", report.resources.real_time_factor),
        ("energy_watt_hours", report.resources.energy_watt_hours),
    ] {
        if value.is_some_and(|number| !number.is_finite() || number < 0.0) {
            return Err(BenchmarkValidationError::InvalidResourceReceipt {
                field: field.into(),
            });
        }
    }
    let expected_report_digest = benchmark_run_report_commitment(report)?;
    if expected_report_digest != report.report_sha256 {
        return Err(BenchmarkValidationError::ReportDigestMismatch);
    }
    Ok(())
}

/// Compare one metric only when the complete protocol fingerprint matches.
/// No cross-track, cross-dataset, cross-renderer or cross-tolerance ranking is
/// possible through this function.
pub fn compare_benchmark_reports(
    left: &BenchmarkRunReportV1,
    right: &BenchmarkRunReportV1,
    metric_id: &str,
) -> Result<BenchmarkComparisonV1, BenchmarkValidationError> {
    validate_benchmark_run_report(left)?;
    validate_benchmark_run_report(right)?;
    if left.protocol_fingerprint != right.protocol_fingerprint {
        return Err(BenchmarkValidationError::IncompatibleProtocol);
    }
    let definition = left
        .protocol
        .metric_definitions
        .iter()
        .find(|definition| definition.metric_id == metric_id)
        .ok_or_else(|| BenchmarkValidationError::MissingComparableMetric {
            metric_id: metric_id.to_string(),
        })?;
    let left_value = overall_metric(left, metric_id)?;
    let right_value = overall_metric(right, metric_id)?;
    let left_overall = left.slices.iter().find(|slice| slice.is_overall)
        .ok_or(BenchmarkValidationError::MissingOverallSlice)?;
    let right_overall = right.slices.iter().find(|slice| slice.is_overall)
        .ok_or(BenchmarkValidationError::MissingOverallSlice)?;
    if left_overall.sample_count != right_overall.sample_count
        || left_value.sample_count != right_value.sample_count
        || left_value.evaluated_sample_manifest_sha256
            != right_value.evaluated_sample_manifest_sha256
    {
        return Err(BenchmarkValidationError::IncompatibleMetricCoverage {
            metric_id: metric_id.to_string(),
        });
    }
    let winner_system_id = match definition.direction {
        MetricDirectionV1::HigherIsBetter if left_value.value > right_value.value => {
            Some(left.system.system_id.clone())
        }
        MetricDirectionV1::LowerIsBetter if left_value.value < right_value.value => {
            Some(left.system.system_id.clone())
        }
        MetricDirectionV1::HigherIsBetter if right_value.value > left_value.value => {
            Some(right.system.system_id.clone())
        }
        MetricDirectionV1::LowerIsBetter if right_value.value < left_value.value => {
            Some(right.system.system_id.clone())
        }
        _ => None,
    };
    Ok(BenchmarkComparisonV1 {
        protocol_fingerprint: left.protocol_fingerprint.clone(),
        metric_id: metric_id.to_string(),
        unit: definition.unit,
        direction: definition.direction,
        left_system_id: left.system.system_id.clone(),
        right_system_id: right.system.system_id.clone(),
        left_value: left_value.value,
        right_value: right_value.value,
        winner_system_id,
    })
}

fn overall_metric<'a>(
    report: &'a BenchmarkRunReportV1,
    metric_id: &str,
) -> Result<&'a MetricValueV1, BenchmarkValidationError> {
    let overall = report
        .slices
        .iter()
        .find(|slice| slice.is_overall)
        .ok_or(BenchmarkValidationError::MissingOverallSlice)?;
    overall
        .metrics
        .iter()
        .find(|metric| metric.metric_id == metric_id)
        .ok_or_else(|| BenchmarkValidationError::MissingComparableMetric {
            metric_id: metric_id.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn protocol(track: EvaluationTrackV1) -> BenchmarkProtocolV1 {
        BenchmarkProtocolV1 {
            schema: MULTI_AXIS_PROTOCOL_SCHEMA_V1.into(),
            protocol_id: "mirex-a2s-2026-blind-fixture-v1".into(),
            benchmark_family: "MIREX-A2S".into(),
            dataset_name: "QuartetsMIREX".into(),
            dataset_version: "2026".into(),
            dataset_manifest_sha256: digest('a'),
            training_corpus_manifest_sha256: digest('b'),
            evaluation_track: track,
            split: GroupSplitV1 {
                claim: GeneralizationClaimV1::OutOfDistribution,
                grouping_unit: GroupingUnitV1::Composer,
                split_manifest_sha256: digest('c'),
                training_group_ids: vec!["Haydn".into(), "Mozart".into()],
                evaluation_group_ids: vec!["Schubert".into(), "Brahms".into()],
                held_out_group_ids: vec!["Schubert".into(), "Brahms".into()],
            },
            annotation: AnnotationProtocolV1 {
                protocol_id: "kern-reference-v1".into(),
                version: "1".into(),
                reference_manifest_sha256: digest('d'),
                reference_count: 1,
                annotator_description: "expert score reference".into(),
            },
            renderer: Some(RendererIdentityV1 {
                renderer_id: "pinned-reference-renderer".into(),
                renderer_version: "1.0".into(),
                soundfont_or_asset_sha256: Some(digest('e')),
                render_configuration_sha256: digest('9'),
                sample_rate_hz: 44_100,
                channel_layout: "stereo".into(),
                loudness_policy: "none".into(),
                tempo_map_policy: "source-score-tempo-map".into(),
            }),
            metric_definitions: vec![
                MetricDefinitionV1 {
                    metric_id: "mv2h".into(),
                    axis: MetricAxisV1::NotationFidelity,
                    direction: MetricDirectionV1::HigherIsBetter,
                    unit: MetricUnitV1::Fraction,
                    protocol_version: "mirex-mv2h-2026-v1".into(),
                    tolerance: Some(MetricToleranceV1 {
                        value: 0.05,
                        unit: "seconds".into(),
                        matching_rule: "official MV2H note onset tolerance".into(),
                    }),
                },
                MetricDefinitionV1 {
                    metric_id: "clews".into(),
                    axis: MetricAxisV1::PlaybackFidelity,
                    direction: MetricDirectionV1::HigherIsBetter,
                    unit: MetricUnitV1::Scalar,
                    protocol_version: "clews-dual-eval-v1".into(),
                    tolerance: None,
                },
            ],
        }
    }

    fn report(system_id: &str, mv2h: f64, track: EvaluationTrackV1) -> BenchmarkRunReportV1 {
        let protocol = protocol(track);
        let protocol_fingerprint = benchmark_protocol_fingerprint(&protocol).unwrap();
        let mut report = BenchmarkRunReportV1 {
            schema: MULTI_AXIS_BENCHMARK_SCHEMA_V1.into(),
            protocol,
            protocol_fingerprint,
            system: SystemIdentityV1 {
                system_id: system_id.into(),
                source_commit: "0123456789abcdef".into(),
                build_sha256: digest('f'),
                model_sha256: Some(digest('1')),
                model_parameter_count: Some(47_100_000),
            },
            source_artifact_sha256: digest('2'),
            model_input_manifest_sha256: digest('3'),
            slices: vec![
                BenchmarkSliceV1 {
                    slice_id: "overall".into(),
                    is_overall: true,
                    grouping_unit: None,
                    grouping_value: None,
                    sample_count: 373,
                    metrics: vec![
                        MetricValueV1 {
                            metric_id: "mv2h".into(),
                            value: mv2h,
                            sample_count: 373,
                            evaluated_sample_manifest_sha256: digest('7'),
                            confidence_interval: Some(ClusteredConfidenceIntervalV1 {
                                lower: (mv2h - 0.02).max(0.0),
                                upper: (mv2h + 0.02).min(1.0),
                                confidence_level: 0.95,
                                cluster_unit: GroupingUnitV1::Composer,
                                cluster_count: 2,
                                method: "cluster bootstrap by composer".into(),
                            }),
                        },
                        MetricValueV1 {
                            metric_id: "clews".into(),
                            value: 0.65,
                            sample_count: 373,
                            evaluated_sample_manifest_sha256: digest('7'),
                            confidence_interval: Some(ClusteredConfidenceIntervalV1 {
                                lower: 0.62,
                                upper: 0.68,
                                confidence_level: 0.95,
                                cluster_unit: GroupingUnitV1::Composer,
                                cluster_count: 2,
                                method: "cluster bootstrap by composer".into(),
                            }),
                        },
                    ],
                },
                BenchmarkSliceV1 {
                    slice_id: "composer-brahms".into(),
                    is_overall: false,
                    grouping_unit: Some(GroupingUnitV1::Composer),
                    grouping_value: Some("Brahms".into()),
                    sample_count: 100,
                    metrics: vec![
                        MetricValueV1 {
                            metric_id: "mv2h".into(),
                            value: mv2h,
                            sample_count: 100,
                            evaluated_sample_manifest_sha256: digest('8'),
                            confidence_interval: None,
                        },
                        MetricValueV1 {
                            metric_id: "clews".into(),
                            value: 0.65,
                            sample_count: 100,
                            evaluated_sample_manifest_sha256: digest('8'),
                            confidence_interval: None,
                        },
                    ],
                },
                BenchmarkSliceV1 {
                    slice_id: "composer-schubert".into(),
                    is_overall: false,
                    grouping_unit: Some(GroupingUnitV1::Composer),
                    grouping_value: Some("Schubert".into()),
                    sample_count: 100,
                    metrics: vec![
                        MetricValueV1 {
                            metric_id: "mv2h".into(),
                            value: mv2h,
                            sample_count: 100,
                            evaluated_sample_manifest_sha256: digest('8'),
                            confidence_interval: None,
                        },
                        MetricValueV1 {
                            metric_id: "clews".into(),
                            value: 0.65,
                            sample_count: 100,
                            evaluated_sample_manifest_sha256: digest('8'),
                            confidence_interval: None,
                        },
                    ],
                },
            ],
            resources: ResourceReceiptV1 {
                inference_milliseconds: Some(1_200.0),
                real_time_factor: Some(0.4),
                peak_memory_bytes: Some(4_000_000_000),
                energy_watt_hours: None,
                hardware_description: "single GPU test fixture".into(),
            },
            report_sha256: String::new(),
        };
        report.report_sha256 = benchmark_run_report_commitment(&report).unwrap();
        report
    }

    #[test]
    fn benchmark_report_commits_protocol_system_and_grouped_metrics() {
        let report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        validate_benchmark_run_report(&report).unwrap();
        assert_eq!(report.report_sha256.len(), 64);
        assert_eq!(report.protocol_fingerprint.len(), 64);
    }

    #[test]
    fn train_evaluation_group_overlap_is_rejected() {
        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.split.evaluation_group_ids.push("Haydn".into());
        protocol.split.held_out_group_ids.push("Haydn".into());
        assert!(matches!(
            validate_benchmark_protocol(&protocol),
            Err(BenchmarkValidationError::TrainingEvaluationOverlap { .. })
        ));
    }

    #[test]
    fn random_excerpt_split_cannot_claim_out_of_distribution_generalization() {
        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.split.grouping_unit = GroupingUnitV1::RandomExcerpt;
        assert_eq!(
            validate_benchmark_protocol(&protocol),
            Err(BenchmarkValidationError::RandomExcerptCannotClaimOutOfDistribution)
        );
    }

    #[test]
    fn playback_metric_requires_pinned_renderer_identity() {
        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.renderer = None;
        assert_eq!(
            validate_benchmark_protocol(&protocol),
            Err(BenchmarkValidationError::PlaybackMetricRequiresRenderer)
        );
    }

    #[test]
    fn comparison_refuses_track_or_protocol_mismatch() {
        let blind = report("blind-system", 0.60, EvaluationTrackV1::Blind);
        let staves = report("staves-system", 0.90, EvaluationTrackV1::StavesInformed);
        assert_eq!(
            compare_benchmark_reports(&blind, &staves, "mv2h"),
            Err(BenchmarkValidationError::IncompatibleProtocol)
        );
    }

    #[test]
    fn matched_comparison_compares_one_metric_without_composite_score() {
        let left = report("system-a", 0.60, EvaluationTrackV1::Blind);
        let right = report("system-b", 0.70, EvaluationTrackV1::Blind);
        let comparison = compare_benchmark_reports(&left, &right, "mv2h").unwrap();
        assert_eq!(comparison.winner_system_id.as_deref(), Some("system-b"));
        assert_eq!(comparison.metric_id, "mv2h");
    }

    #[test]
    fn changed_report_or_protocol_is_rejected() {
        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.slices[0].metrics[0].value = 0.99;
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::ReportDigestMismatch)
        );

        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.protocol.dataset_name = "different dataset".into();
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::ProtocolFingerprintMismatch)
        );
    }

    #[test]
    fn fraction_metric_outside_unit_interval_is_rejected() {
        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.slices[0].metrics[0].value = 1.2;
        report.report_sha256 = benchmark_run_report_commitment(&report).unwrap();
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::InvalidMetricValue {
                metric_id: "mv2h".into()
            })
        );
    }

    #[test]
    fn comparison_rejects_different_abstention_or_sample_subsets() {
        let left = report("system-a", 0.60, EvaluationTrackV1::Blind);
        let mut right = report("system-b", 0.70, EvaluationTrackV1::Blind);
        right.slices[0].metrics[0].evaluated_sample_manifest_sha256 = digest('4');
        right.report_sha256 = benchmark_run_report_commitment(&right).unwrap();
        assert_eq!(
            compare_benchmark_reports(&left, &right, "mv2h"),
            Err(BenchmarkValidationError::IncompatibleMetricCoverage {
                metric_id: "mv2h".into()
            })
        );
    }


    #[test]
    fn mixed_ood_evaluation_groups_cannot_be_labeled_fully_held_out() {
        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.split.evaluation_group_ids.push("Haydn".into());
        assert_eq!(
            validate_benchmark_protocol(&protocol),
            Err(BenchmarkValidationError::TrainingEvaluationOverlap {
                group_id: "Haydn".into()
            })
        );

        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.split.evaluation_group_ids.push("Tchaikovsky".into());
        assert_eq!(
            validate_benchmark_protocol(&protocol),
            Err(BenchmarkValidationError::InvalidSplitPolicy)
        );
    }

    #[test]
    fn in_distribution_claim_allows_work_or_session_holdouts_without_composer_leakage_claims() {
        let mut protocol = protocol(EvaluationTrackV1::Blind);
        protocol.split.claim = GeneralizationClaimV1::InDistribution;
        protocol.split.grouping_unit = GroupingUnitV1::Work;
        protocol.split.training_group_ids = vec!["work-a".into(), "work-b".into()];
        protocol.split.evaluation_group_ids = vec!["work-c".into()];
        protocol.split.held_out_group_ids.clear();
        assert_eq!(validate_benchmark_protocol(&protocol), Ok(()));

        protocol.split.grouping_unit = GroupingUnitV1::RecordingSession;
        assert_eq!(validate_benchmark_protocol(&protocol), Ok(()));
    }

    #[test]
    fn ood_report_requires_every_held_out_group_and_metric_slice() {
        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.slices.remove(1);
        report.report_sha256 = benchmark_run_report_commitment(&report).unwrap();
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::MissingGroupSlice {
                group_id: "Brahms".into()
            })
        );

        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.slices[1].metrics.pop();
        report.report_sha256 = benchmark_run_report_commitment(&report).unwrap();
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::MissingGroupMetricCoverage {
                group_id: "Brahms".into(),
                metric_id: "mv2h".into()
            })
        );
    }

    #[test]
    fn overall_slice_must_report_every_declared_metric() {
        let mut report = report("system-a", 0.61, EvaluationTrackV1::Blind);
        report.slices[0].metrics.pop();
        report.report_sha256 = benchmark_run_report_commitment(&report).unwrap();
        assert_eq!(
            validate_benchmark_run_report(&report),
            Err(BenchmarkValidationError::MissingMetricCoverage {
                metric_id: "clews".into()
            })
        );
    }

}
