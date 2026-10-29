// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed semantic validation for versioned symbolic-to-audio edit contracts.
//!
//! This module validates contract meaning after deserialization. JSON Schema (or
//! an equivalent structural validator) is still required at the boundary: a
//! missing required field fails deserialization before this validator runs.
//! A valid contract is not evidence that an edit was performed successfully.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

pub const EDIT_CONTRACT_SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditContract {
    pub schema_version: String,
    pub contract_id: String,
    pub source: SourceAsset,
    pub authorization_mask: Vec<RegionMask>,
    pub requested_changes: Vec<RequestedChange>,
    pub protected_invariants: Vec<ProtectedInvariant>,
    pub measurement: Measurement,
    pub abstention_policy: AbstentionPolicy,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAsset {
    pub symbolic_sha256: String,
    pub audio_sha256: String,
    pub audio_format: String,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub duration_frames: u64,
    pub track_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionMask {
    pub region_id: String,
    pub start_us: u64,
    pub end_us: u64,
    pub track_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedChange {
    pub axis: PropertyAxis,
    pub operation: EditOperation,
    pub region_ids: Vec<String>,
    pub target: ContractValue,
    pub tolerance: Option<Tolerance>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedInvariant {
    pub axis: PropertyAxis,
    pub scope: InvariantScope,
    pub region_ids: Option<Vec<String>>,
    pub track_ids: Option<Vec<String>>,
    pub tolerance: Tolerance,
    pub critical: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    pub metric: ToleranceMetric,
    pub max_deviation: f64,
    pub unit: Option<String>,
    pub calibration_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ContractValue {
    Bool(bool),
    Number(f64),
    String(String),
    Object(BTreeMap<String, ContractValue>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyAxis {
    Pitch,
    Onset,
    Duration,
    Velocity,
    NoteCount,
    Instrumentation,
    Timbre,
    Dynamics,
    StereoImage,
    AccompanimentDensity,
    Harmony,
    Rhythm,
    Structure,
    Transients,
    SpectralEnvelope,
    SourceAttribution,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOperation {
    TransposeSemitones,
    SetPitch,
    ScaleDuration,
    ScaleOnset,
    SetVelocity,
    ChangeInstrument,
    ChangeDynamics,
    ChangeStereo,
    ChangeDensity,
    ChangeHarmony,
    ChangeRhythm,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvariantScope {
    OutsideMask,
    InsideMask,
    Everywhere,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToleranceMetric {
    Absolute,
    Relative,
    LogRatio,
    FeatureDistance,
    CategoricalExact,
    AnnotatorCalibrated,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub renderer: RendererSpec,
    pub verifier: VerifierSpec,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RendererSpec {
    pub renderer_id: String,
    pub renderer_version: String,
    pub instrument_bank_sha256: Option<String>,
    pub effect_chain_sha256: Option<String>,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub seed: Option<u64>,
    pub deterministic: bool,
    pub latency_compensation_frames: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierSpec {
    pub symbolic_verifier_version: String,
    pub acoustic_pipeline_version: String,
    pub config_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbstentionPolicy {
    pub required_evidence: Vec<RequiredEvidence>,
    pub on_detector_out_of_domain: String,
    pub on_measurement_uncertainty: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredEvidence {
    Symbolic,
    Acoustic,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractStatus {
    Valid,
    Reject,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractIssue {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractValidationReport {
    pub status: ContractStatus,
    pub issues: Vec<ContractIssue>,
}

impl ContractValidationReport {
    fn finish(issues: Vec<ContractIssue>) -> Self {
        let has_rejection = issues.iter().any(|issue| {
            !matches!(
                issue.code,
                "UNSUPPORTED_OPERATION" | "REPLAY_NOT_REPRODUCIBLE"
            )
        });
        let status = if has_rejection {
            ContractStatus::Reject
        } else if issues.is_empty() {
            ContractStatus::Valid
        } else {
            ContractStatus::Abstain
        };
        Self { status, issues }
    }
}

/// Validate semantic invariants which JSON Schema alone cannot express.
///
/// Source duration is defined by integer frame count and sample rate. Time
/// comparisons use checked-width integer arithmetic; no float conversion or
/// rounded microsecond duration decides whether a mask is in bounds.
pub fn validate_edit_contract(contract: &EditContract) -> ContractValidationReport {
    let mut issues = Vec::new();
    let mut codes = BTreeSet::new();
    macro_rules! add_issue {
        ($code:expr, $message:expr) => {{
            let issue_code: &'static str = $code;
            if codes.insert(issue_code) {
                issues.push(ContractIssue {
                    code: issue_code,
                    message: $message,
                });
            }
        }};
    }

    if contract.schema_version != EDIT_CONTRACT_SCHEMA_VERSION {
        add_issue!(
            "UNSUPPORTED_VERSION",
            format!(
                "schema version {} is not supported by this validator",
                contract.schema_version
            ),
        );
    }
    if !valid_contract_id(&contract.contract_id) {
        add_issue!(
            "INVALID_CONTRACT_ID",
            "contract_id must use the versioned MEL-MR-EVAL-005 identifier form".to_owned(),
        );
    }

    let source = &contract.source;
    if !is_sha256(&source.symbolic_sha256)
        || !is_sha256(&source.audio_sha256)
        || !is_sha256(&contract.measurement.verifier.config_sha256)
        || source.audio_format.trim().is_empty()
        || !(8_000..=768_000).contains(&source.sample_rate_hz)
        || source.channel_count == 0
        || source.channel_count > 64
        || source.duration_frames == 0
        || source.track_ids.is_empty()
        || source.track_ids.iter().any(|id| id.trim().is_empty())
    {
        add_issue!(
            "SCHEMA_REJECT",
            "source metadata is missing, malformed, or outside the declared domain".to_owned(),
        );
    }
    if source
        .track_ids
        .iter()
        .any(|id| id.trim().is_empty())
        || duplicate_strings(&source.track_ids)
    {
        add_issue!(
            "DUPLICATE_TRACK_ID",
            "source track identifiers must be non-empty and unique".to_owned(),
        );
    }

    let source_tracks: BTreeSet<String> = source.track_ids.iter().cloned().collect();
    let mut region_by_id: BTreeMap<String, &RegionMask> = BTreeMap::new();

    for region in &contract.authorization_mask {
        if region.region_id.trim().is_empty()
            || region.track_ids.is_empty()
            || region.track_ids.iter().any(|id| id.trim().is_empty())
        {
            add_issue!(
                "SCHEMA_REJECT",
                format!("region {} has an empty identifier or track set", region.region_id),
            );
        }
        if region_by_id.insert(region.region_id.clone(), region).is_some() {
            add_issue!(
                "DUPLICATE_REGION_ID",
                format!("region ID {} is declared more than once", region.region_id),
            );
        }
        if region.end_us <= region.start_us {
            add_issue!(
                "INVALID_REGION_INTERVAL",
                format!("region {} must have end_us > start_us", region.region_id),
            );
        }
        // Exact inequality: end_us * sample_rate <= duration_frames * 1,000,000.
        let left = u128::from(region.end_us) * u128::from(source.sample_rate_hz);
        let right = u128::from(source.duration_frames) * 1_000_000_u128;
        if left > right {
            add_issue!(
                "REGION_OUT_OF_BOUNDS",
                format!("region {} extends beyond source audio duration", region.region_id),
            );
        }
        for track in &region.track_ids {
            if !source_tracks.contains(track) {
                add_issue!(
                    "UNKNOWN_TRACK_ID",
                    format!("region {} references unknown track {}", region.region_id, track),
                );
            }
        }
    }

    if contract.authorization_mask.is_empty()
        || contract.requested_changes.is_empty()
        || contract.protected_invariants.is_empty()
    {
        add_issue!(
            "SCHEMA_REJECT",
            "mask, requested changes, and protected invariants must not be empty".to_owned(),
        );
    }

    for i in 0..contract.authorization_mask.len() {
        for j in (i + 1)..contract.authorization_mask.len() {
            let left = &contract.authorization_mask[i];
            let right = &contract.authorization_mask[j];
            let shares_track = left
                .track_ids
                .iter()
                .any(|track| right.track_ids.contains(track));
            let overlaps = left.start_us < right.end_us && right.start_us < left.end_us;
            if shares_track && overlaps {
                add_issue!(
                    "AMBIGUOUS_OVERLAPPING_MASKS",
                    format!(
                        "regions {} and {} overlap on at least one track",
                        left.region_id, right.region_id
                    ),
                );
            }
        }
    }

    for change in &contract.requested_changes {
        for region_id in &change.region_ids {
            if !region_by_id.contains_key(region_id) {
                add_issue!(
                    "UNKNOWN_REGION_ID",
                    format!("requested change references unknown region {}", region_id),
                );
            }
        }

        match operation_axis(change.operation) {
            Some(axis) if axis != change.axis => push(
                "OPERATION_AXIS_MISMATCH",
                format!(
                    "operation {:?} changes {:?}, not {:?}",
                    change.operation, axis, change.axis
                ),
            ),
            None => push(
                "UNSUPPORTED_OPERATION",
                format!("operation {:?} has no supported semantics", change.operation),
            ),
            _ => {}
        }

        match validate_target(change.operation, &change.target) {
            TargetCheck::Ok => {}
            TargetCheck::TypeMismatch => push(
                "TARGET_TYPE_MISMATCH",
                format!("target type is invalid for operation {:?}", change.operation),
            ),
            TargetCheck::OutOfDomain => push(
                "TARGET_OUT_OF_DOMAIN",
                format!("target value is outside the supported domain for {:?}", change.operation),
            ),
            TargetCheck::NoOp => push(
                "NOOP_REQUEST",
                format!("operation {:?} requests a no-op", change.operation),
            ),
            TargetCheck::Unsupported => push(
                "UNSUPPORTED_OPERATION",
                format!("operation {:?} is not implemented by this contract validator", change.operation),
            ),
        }
    }

    for i in 0..contract.requested_changes.len() {
        for j in (i + 1)..contract.requested_changes.len() {
            let left = &contract.requested_changes[i];
            let right = &contract.requested_changes[j];
            if left.axis == right.axis
                && left.region_ids.iter().any(|id| right.region_ids.contains(id))
            {
                add_issue!(
                    "AMBIGUOUS_OVERLAPPING_REQUESTS",
                    format!(
                        "requested changes {} and {} overlap on axis {:?} without composition order",
                        i, j, left.axis
                    ),
                );
            }
        }
    }

    for invariant in &contract.protected_invariants {
        if !invariant.tolerance.max_deviation.is_finite()
            || invariant.tolerance.max_deviation < 0.0
        {
            add_issue!(
                "TOLERANCE_OUT_OF_DOMAIN",
                "tolerance max_deviation must be finite and non-negative".to_owned(),
            );
        }
        if let Some(region_ids) = &invariant.region_ids {
            for region_id in region_ids {
                if !region_by_id.contains_key(region_id.as_str()) {
                    add_issue!(
                        "UNKNOWN_REGION_ID",
                        format!("protected invariant references unknown region {}", region_id),
                    );
                }
            }
        }
        if let Some(track_ids) = &invariant.track_ids {
            for track_id in track_ids {
                if !source_tracks.contains(track_id) {
                    add_issue!(
                        "UNKNOWN_TRACK_ID",
                        format!("protected invariant references unknown track {}", track_id),
                    );
                }
            }
        }

        let tolerance = &invariant.tolerance;
        let missing_calibration = tolerance
            .calibration_id
            .as_deref()
            .is_none_or(|id| id.trim().is_empty());
        if matches!(
            tolerance.metric,
            ToleranceMetric::FeatureDistance | ToleranceMetric::AnnotatorCalibrated
        ) && missing_calibration
        {
            add_issue!(
                "CALIBRATION_ID_REQUIRED",
                format!("axis {:?} uses a calibrated metric without calibration_id", invariant.axis),
            );
            if tolerance.metric == ToleranceMetric::AnnotatorCalibrated {
                add_issue!(
                    "SCHEMA_REJECT",
                    "annotator_calibrated requires calibration_id by schema".to_owned(),
                );
            }
        }
        if !compatible_unit(invariant.axis, tolerance.unit.as_deref()) {
            add_issue!(
                "TOLERANCE_UNIT_AXIS_MISMATCH",
                format!("unit {:?} is incompatible with axis {:?}", tolerance.unit, invariant.axis),
            );
        }

        for change in &contract.requested_changes {
            if change.axis != invariant.axis {
                continue;
            }
            let scope_overlaps = match invariant.scope {
                InvariantScope::Everywhere => true,
                InvariantScope::OutsideMask => false,
                InvariantScope::InsideMask => invariant
                    .region_ids
                    .as_ref()
                    .is_none_or(|ids| ids.iter().any(|id| change.region_ids.contains(id))),
            };
            if scope_overlaps {
                add_issue!(
                    "REQUEST_PROTECTED_SCOPE_CONFLICT",
                    format!(
                        "requested {:?} change conflicts with a {:?} invariant",
                        change.axis, invariant.scope
                    ),
                );
            }
        }
    }

    let required: BTreeSet<RequiredEvidenceKey> = contract
        .abstention_policy
        .required_evidence
        .iter()
        .map(|e| match e {
            RequiredEvidence::Symbolic => RequiredEvidenceKey::Symbolic,
            RequiredEvidence::Acoustic => RequiredEvidenceKey::Acoustic,
            RequiredEvidence::Replay => RequiredEvidenceKey::Replay,
        })
        .collect();
    if required.len() != 3 {
        add_issue!(
            "SCHEMA_REJECT",
            "required evidence must include symbolic, acoustic, and replay exactly once".to_owned(),
        );
    }
    if contract.abstention_policy.on_detector_out_of_domain != "abstain"
        || contract.abstention_policy.on_measurement_uncertainty != "abstain"
    {
        add_issue!(
            "SCHEMA_REJECT",
            "out-of-domain and uncertain measurement policies must fail closed to abstain".to_owned(),
        );
    }
    let renderer = &contract.measurement.renderer;
    if renderer.renderer_id.trim().is_empty()
        || renderer.renderer_version.trim().is_empty()
        || contract.measurement.verifier.symbolic_verifier_version.trim().is_empty()
        || contract.measurement.verifier.acoustic_pipeline_version.trim().is_empty()
        || !(8_000..=768_000).contains(&renderer.sample_rate_hz)
        || renderer.channel_count == 0
        || renderer.channel_count > 64
        || renderer.instrument_bank_sha256.as_deref().is_some_and(|v| !is_sha256(v))
        || renderer.effect_chain_sha256.as_deref().is_some_and(|v| !is_sha256(v))
    {
        add_issue!(
            "SCHEMA_REJECT",
            "renderer / verifier identity or output format is malformed".to_owned(),
        );
    }
    if !renderer.deterministic && required.contains(&RequiredEvidenceKey::Replay) {
        add_issue!(
            "REPLAY_NOT_REPRODUCIBLE",
            "deterministic replay is required but renderer is declared nondeterministic".to_owned(),
        );
    }

    ContractValidationReport::finish(issues)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RequiredEvidenceKey {
    Symbolic,
    Acoustic,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetCheck {
    Ok,
    TypeMismatch,
    OutOfDomain,
    NoOp,
    Unsupported,
}

fn operation_axis(operation: EditOperation) -> Option<PropertyAxis> {
    match operation {
        EditOperation::TransposeSemitones | EditOperation::SetPitch => Some(PropertyAxis::Pitch),
        EditOperation::ScaleDuration => Some(PropertyAxis::Duration),
        EditOperation::ScaleOnset => Some(PropertyAxis::Onset),
        EditOperation::SetVelocity => Some(PropertyAxis::Velocity),
        EditOperation::ChangeInstrument => Some(PropertyAxis::Instrumentation),
        EditOperation::ChangeDynamics => Some(PropertyAxis::Dynamics),
        EditOperation::ChangeStereo => Some(PropertyAxis::StereoImage),
        EditOperation::ChangeDensity => Some(PropertyAxis::AccompanimentDensity),
        EditOperation::ChangeHarmony => Some(PropertyAxis::Harmony),
        EditOperation::ChangeRhythm => Some(PropertyAxis::Rhythm),
        EditOperation::Other => None,
    }
}

fn validate_target(operation: EditOperation, target: &ContractValue) -> TargetCheck {
    let number = match target {
        ContractValue::Number(value) if value.is_finite() => Some(*value),
        ContractValue::Number(_) => return TargetCheck::OutOfDomain,
        _ => None,
    };
    match operation {
        EditOperation::TransposeSemitones => match number {
            None => TargetCheck::TypeMismatch,
            Some(value) if value.fract() != 0.0 || value.abs() > 48.0 => TargetCheck::OutOfDomain,
            Some(0.0) => TargetCheck::NoOp,
            Some(_) => TargetCheck::Ok,
        },
        EditOperation::SetPitch => match number {
            None => TargetCheck::TypeMismatch,
            Some(value) if value.fract() != 0.0 || !(0.0..=127.0).contains(&value) => {
                TargetCheck::OutOfDomain
            }
            Some(_) => TargetCheck::Ok,
        },
        EditOperation::ScaleDuration | EditOperation::ScaleOnset => match number {
            None => TargetCheck::TypeMismatch,
            Some(value) if value <= 0.0 => TargetCheck::OutOfDomain,
            Some(1.0) => TargetCheck::NoOp,
            Some(_) => TargetCheck::Ok,
        },
        EditOperation::SetVelocity => match number {
            None => TargetCheck::TypeMismatch,
            Some(value) if value.fract() != 0.0 || !(1.0..=127.0).contains(&value) => {
                TargetCheck::OutOfDomain
            }
            Some(_) => TargetCheck::Ok,
        },
        EditOperation::Other => TargetCheck::Unsupported,
        // These operation axes are enumerated by the draft schema, but their
        // target domains are not frozen yet; abstain until they are specified.
        EditOperation::ChangeInstrument
        | EditOperation::ChangeDynamics
        | EditOperation::ChangeStereo
        | EditOperation::ChangeDensity
        | EditOperation::ChangeHarmony
        | EditOperation::ChangeRhythm => TargetCheck::Unsupported,
    }
}

fn compatible_unit(axis: PropertyAxis, unit: Option<&str>) -> bool {
    let Some(unit) = unit else {
        return false;
    };
    match axis {
        PropertyAxis::Pitch => matches!(unit, "semitones" | "MIDI-note"),
        PropertyAxis::Onset | PropertyAxis::Duration => {
            matches!(unit, "microseconds" | "milliseconds")
        }
        PropertyAxis::NoteCount => unit == "events",
        PropertyAxis::Velocity => unit == "MIDI-velocity",
        PropertyAxis::Timbre | PropertyAxis::SpectralEnvelope => {
            unit == "normalized-feature-distance"
        }
        PropertyAxis::StereoImage => unit == "normalized-stereo-position",
        PropertyAxis::Dynamics => {
            matches!(unit, "normalized-feature-distance" | "decibels" | "MIDI-velocity")
        }
        PropertyAxis::AccompanimentDensity => matches!(unit, "events-per-beat" | "relative"),
        PropertyAxis::Harmony | PropertyAxis::Rhythm | PropertyAxis::Structure => {
            matches!(unit, "categorical" | "normalized-feature-distance")
        }
        PropertyAxis::Instrumentation | PropertyAxis::SourceAttribution => {
            matches!(unit, "categorical" | "normalized-feature-distance")
        }
        PropertyAxis::Transients => {
            matches!(unit, "microseconds" | "milliseconds" | "normalized-feature-distance")
        }
        PropertyAxis::Other => unit == "normalized-feature-distance",
    }
}

fn valid_contract_id(id: &str) -> bool {
    id.starts_with("MEL-MR-EVAL-005-")
        && id.len() >= 20
        && id
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-')
        && id.chars().last().is_some_and(|c| c.is_ascii_alphanumeric())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn duplicate_strings(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(value.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const EXAMPLE: &str = include_str!("../mel-mr-eval-005-contract-example-v0.1.json");
    const VECTORS: &str =
        include_str!("../mel-mr-eval-005-semantic-validation-cases-v0.1.jsonl");

    fn parse_contract(value: Value) -> Result<EditContract, ContractValidationReport> {
        match serde_json::from_value::<EditContract>(value) {
            Ok(contract) => Ok(contract),
            Err(error) => Err(ContractValidationReport {
                status: ContractStatus::Reject,
                issues: vec![ContractIssue {
                    code: "SCHEMA_REJECT",
                    message: error.to_string(),
                }],
            }),
        }
    }

    fn apply_mutation(document: &mut Value, mutation: &Value) {
        let op = mutation["op"].as_str().expect("mutation op");
        let path = mutation["path"].as_str().expect("mutation path");
        let (parent_path, raw_key) = path.rsplit_once('/').expect("JSON pointer parent/key");
        let key = raw_key.replace("~1", "/").replace("~0", "~");
        let parent = document.pointer_mut(parent_path).expect("mutation parent pointer");
        match parent {
            Value::Object(map) => match op {
                "add" | "replace" => {
                    map.insert(key, mutation["value"].clone());
                }
                "remove" => {
                    map.remove(&key).expect("removing existing object property");
                }
                _ => panic!("unsupported mutation op"),
            },
            Value::Array(items) => {
                if op == "add" && key == "-" {
                    items.push(mutation["value"].clone());
                    return;
                }
                let index: usize = key.parse().expect("array index");
                match op {
                    "add" => items.insert(index, mutation["value"].clone()),
                    "replace" => items[index] = mutation["value"].clone(),
                    "remove" => {
                        items.remove(index);
                    }
                    _ => panic!("unsupported mutation op"),
                }
            }
            _ => panic!("mutation parent must be object or array"),
        }
    }

    fn status_name(status: ContractStatus) -> &'static str {
        match status {
            ContractStatus::Valid => "valid",
            ContractStatus::Reject => "reject",
            ContractStatus::Abstain => "abstain",
        }
    }

    #[test]
    fn base_contract_is_semantically_valid_but_placeholder_only() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let contract = parse_contract(value).expect("typed contract parses");
        let report = validate_edit_contract(&contract);
        assert_eq!(report.status, ContractStatus::Valid, "{report:?}");
        assert!(
            contract.notes.as_deref().unwrap_or("").contains("placeholders"),
            "the example must remain visibly non-evidentiary"
        );
    }

    #[test]
    fn semantic_validation_vectors_match_declared_verdicts_and_codes() {
        let base: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut cases = 0usize;
        for line in VECTORS.lines().filter(|line| !line.trim().is_empty()) {
            let vector: Value = serde_json::from_str(line).expect("valid JSONL vector");
            let mut mutated = base.clone();
            for mutation in vector["mutations"].as_array().expect("mutations array") {
                apply_mutation(&mut mutated, mutation);
            }

            let report = match parse_contract(mutated) {
                Ok(contract) => validate_edit_contract(&contract),
                Err(report) => report,
            };
            let expected_status = vector["verdict"].as_str().expect("expected verdict");
            assert_eq!(
                status_name(report.status),
                expected_status,
                "case {}: {:?}",
                vector["case_id"],
                report.issues
            );
            let expected_codes: Vec<&str> = vector["expected_codes"]
                .as_array()
                .expect("expected code array")
                .iter()
                .map(|value| value.as_str().expect("code string"))
                .collect();
            for expected in expected_codes {
                assert!(
                    report.issues.iter().any(|issue| issue.code == expected),
                    "case {} missing expected code {} (actual {:?})",
                    vector["case_id"],
                    expected,
                    report.issues
                );
            }
            cases += 1;
        }
        assert_eq!(cases, 25, "the published v0.1 corpus must be complete");
    }

    #[test]
    fn duration_bounds_use_integer_arithmetic() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut contract = parse_contract(value).expect("typed contract parses");
        contract.authorization_mask[0].end_us = 10_000_001;
        let report = validate_edit_contract(&contract);
        assert!(report.issues.iter().any(|issue| issue.code == "REGION_OUT_OF_BOUNDS"));
    }

    #[test]
    fn unknown_schema_version_fails_closed() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut contract = parse_contract(value).expect("typed contract parses");
        contract.schema_version = "0.2.0".to_owned();
        assert_eq!(
            validate_edit_contract(&contract).status,
            ContractStatus::Reject
        );
    }

    #[test]
    fn nondeterminism_abstains_when_replay_is_required() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut contract = parse_contract(value).expect("typed contract parses");
        contract.measurement.renderer.deterministic = false;
        let report = validate_edit_contract(&contract);
        assert_eq!(report.status, ContractStatus::Abstain);
        assert!(report.issues.iter().any(|i| i.code == "REPLAY_NOT_REPRODUCIBLE"));
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut contract = parse_contract(value).expect("typed contract parses");
        contract.source.audio_sha256 = "not-a-sha256".to_owned();
        assert_eq!(validate_edit_contract(&contract).status, ContractStatus::Reject);
    }

    #[test]
    fn operation_targets_are_typed_and_no_ops_are_rejected() {
        let value: Value = serde_json::from_str(EXAMPLE).expect("valid example JSON");
        let mut contract = parse_contract(value).expect("typed contract parses");
        contract.requested_changes[0].target = ContractValue::Number(0.0);
        let report = validate_edit_contract(&contract);
        assert!(report.issues.iter().any(|i| i.code == "NOOP_REQUEST"));
    }
}
