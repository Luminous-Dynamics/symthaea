#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Grounded natural-language music-edit plan contract.
//!
//! This crate verifies a planner's structured output against a frozen source
//! observation. It does not parse natural language, perform audio analysis,
//! render audio, or authorize a transaction. Ambiguous and unsupported plans
//! are non-executable by construction.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SOURCE_OBSERVATION_SCHEMA_V1: &str = "mel-music-source-observation-v1";
pub const INTENT_PLAN_SCHEMA_V1: &str = "mel-music-edit-intent-plan-v1";
pub const MAX_SOURCE_EVENTS_V1: usize = 65_536;
pub const MAX_SOURCE_OCCURRENCES_V1: usize = 16_384;
pub const MAX_SOURCE_ANCHORS_V1: usize = 8_192;
pub const MAX_SELECTED_OCCURRENCES_V1: usize = 1_024;
pub const MAX_CONFIDENCE_BP_V1: u16 = 10_000;
const OBSERVATION_DOMAIN_V1: &[u8] = b"luminous.mel.music-source-observation.v1";
const PLAN_DOMAIN_V1: &[u8] = b"luminous.mel.music-edit-intent-plan.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceEventV1 {
    pub event_id: String,
    pub part_id: String,
    pub onset_ticks: u64,
    pub duration_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAnchorKindV1 {
    Section,
    Bar,
    Beat,
    Phrase,
    Event,
    ExplicitTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalAnchorV1 {
    pub anchor_id: String,
    pub kind: TemporalAnchorKindV1,
    pub label: String,
    pub start_tick: u64,
    pub end_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceOccurrenceV1 {
    pub occurrence_id: String,
    pub semantic_class_id: String,
    pub start_tick: u64,
    pub end_tick: u64,
    /// Stable event IDs grounded by this occurrence observation.
    pub event_ids: Vec<String>,
    /// Stable source anchors that contain or identify this occurrence.
    pub anchor_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceObservationV1 {
    pub schema_version: String,
    pub source_artifact_sha256: String,
    pub source_lineage_sha256: String,
    /// Digest of the upstream observation/grounding attestation receipt.
    pub observation_evidence_receipt_sha256: String,
    pub duration_ticks: u64,
    /// Canonical order by event_id.
    pub events: Vec<SourceEventV1>,
    /// Canonical order by anchor_id.
    pub anchors: Vec<TemporalAnchorV1>,
    /// Canonical order by occurrence_id.
    pub occurrences: Vec<SourceOccurrenceV1>,
    pub observation_sha256: String,
}

#[derive(Serialize)]
struct SourceObservationCommitmentV1<'a> {
    schema_version: &'a str,
    source_artifact_sha256: &'a str,
    source_lineage_sha256: &'a str,
    observation_evidence_receipt_sha256: &'a str,
    duration_ticks: u64,
    events: &'a [SourceEventV1],
    anchors: &'a [TemporalAnchorV1],
    occurrences: &'a [SourceOccurrenceV1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditOperationV1 {
    Add,
    Remove,
    Replace,
    AdjustGain,
    Transpose,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OccurrenceSelectorV1 {
    UniqueMatching {
        semantic_class_id: String,
    },
    NthMatching {
        semantic_class_id: String,
        /// One-based ordinal after chronological ordering.
        ordinal: u16,
    },
    AllMatching {
        semantic_class_id: String,
    },
    WithinAnchor {
        semantic_class_id: String,
        anchor_id: String,
        relation: AnchorRelationV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorRelationV1 {
    Inside,
    After,
    Before,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditParameterValueV1 {
    Scalar {
        integer_value: i64,
        unit: String,
    },
    Identity {
        value: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditParameterV1 {
    pub parameter_id: String,
    pub value: EditParameterValueV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanDispositionV1 {
    Ready,
    NeedsClarification,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundingEvidenceKindV1 {
    EventMap,
    TemporalAnchor,
    HumanAnnotation,
    AudioModelObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccurrenceGroundingEvidenceV1 {
    pub occurrence_id: String,
    pub source_observation_sha256: String,
    /// Digest of the occurrence-level detector/annotation evidence receipt.
    pub evidence_receipt_sha256: String,
    pub evidence_kind: GroundingEvidenceKindV1,
    pub confidence_bp: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntentTargetWindowV1 {
    pub occurrence_id: String,
    pub start_tick: u64,
    pub end_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditIntentPlanV1 {
    pub schema_version: String,
    pub plan_id: String,
    pub instruction_sha256: String,
    pub source_observation_sha256: String,
    pub disposition: PlanDispositionV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<MusicEditOperationV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<OccurrenceSelectorV1>,
    /// Canonical sorted occurrence IDs selected by a Ready plan.
    pub selected_occurrence_ids: Vec<String>,
    /// Canonical sorted event IDs selected by a Ready plan.
    pub selected_event_ids: Vec<String>,
    /// Exact complement of selected_event_ids in the frozen source inventory.
    pub protected_event_ids: Vec<String>,
    /// One half-open window per selected occurrence.
    pub target_windows: Vec<IntentTargetWindowV1>,
    /// Canonical order by parameter_id.
    pub parameters: Vec<EditParameterV1>,
    /// One independently bound observation per selected occurrence.
    pub grounding_evidence: Vec<OccurrenceGroundingEvidenceV1>,
    /// Non-executable clarification candidate set for ambiguous plans.
    pub clarification_candidates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clarification_question: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention_reason: Option<String>,
    /// Reported confidence; never bypasses any structural or evidence check.
    pub confidence_bp: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentPlanIssueV1 {
    InvalidObservationSchema,
    InvalidPlanSchema,
    InvalidDigest,
    EmptyIdentity,
    InvalidDuration,
    NonCanonicalEventOrder,
    DuplicateEventId,
    InvalidEvent,
    NonCanonicalAnchorOrder,
    DuplicateAnchorId,
    InvalidAnchor,
    NonCanonicalOccurrenceOrder,
    DuplicateOccurrenceId,
    InvalidOccurrence,
    UnknownEventReference,
    EventOutsideOccurrenceWindow,
    UnknownAnchorReference,
    OccurrenceOutsideAnchor,
    NonCanonicalReferenceOrder,
    ObservationDigestMismatch,
    SourceBindingMismatch,
    ConfidenceOutOfRange,
    InvalidSelector,
    InvalidDispositionShape,
    SelectionSetMismatch,
    EventSupportMismatch,
    ProtectedComplementMismatch,
    TargetWindowMismatch,
    GroundingEvidenceMismatch,
    InvalidParameterOrder,
    DuplicateParameter,
    InvalidOperationParameter,
    ClarificationCandidateMismatch,
    UnexpectedExecutableFields,
}

impl fmt::Display for IntentPlanIssueV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidObservationSchema => "source observation schema is invalid",
            Self::InvalidPlanSchema => "intent plan schema is invalid",
            Self::InvalidDigest => "a required SHA-256 digest is malformed",
            Self::EmptyIdentity => "a required stable identity is empty",
            Self::InvalidDuration => "source or interval duration is invalid",
            Self::NonCanonicalEventOrder => "source events must be sorted by event_id",
            Self::DuplicateEventId => "source event IDs must be unique",
            Self::InvalidEvent => "a source event is invalid or outside source duration",
            Self::NonCanonicalAnchorOrder => "anchors must be sorted by anchor_id",
            Self::DuplicateAnchorId => "anchor IDs must be unique",
            Self::InvalidAnchor => "a temporal anchor is invalid or outside source duration",
            Self::NonCanonicalOccurrenceOrder => "occurrences must be sorted by occurrence_id",
            Self::DuplicateOccurrenceId => "occurrence IDs must be unique",
            Self::InvalidOccurrence => "a source occurrence has invalid interval or identity",
            Self::UnknownEventReference => "an occurrence references an event absent from the source",
            Self::EventOutsideOccurrenceWindow => "a referenced event falls outside its occurrence interval",
            Self::UnknownAnchorReference => "an occurrence references an anchor absent from the source",
            Self::OccurrenceOutsideAnchor => "an occurrence falls outside a referenced temporal anchor",
            Self::NonCanonicalReferenceOrder => "event and anchor references must be sorted and unique",
            Self::ObservationDigestMismatch => "source observation commitment does not match its contents",
            Self::SourceBindingMismatch => "plan is bound to a different source observation",
            Self::ConfidenceOutOfRange => "confidence is outside [0, 10000]",
            Self::InvalidSelector => "occurrence selector is invalid or references an unknown anchor",
            Self::InvalidDispositionShape => "plan disposition has an invalid executable/clarification shape",
            Self::SelectionSetMismatch => "selected occurrences do not match the frozen selector semantics",
            Self::EventSupportMismatch => "selected event IDs differ from the grounded occurrence support",
            Self::ProtectedComplementMismatch => "protected event set is not the exact source complement",
            Self::TargetWindowMismatch => "target windows differ from exact occurrence boundaries",
            Self::GroundingEvidenceMismatch => "grounding evidence does not bind every selected occurrence exactly",
            Self::InvalidParameterOrder => "parameters must be sorted by parameter_id",
            Self::DuplicateParameter => "parameter IDs must be unique",
            Self::InvalidOperationParameter => "operation parameters or units are invalid",
            Self::ClarificationCandidateMismatch => "clarification candidates differ from the observed ambiguous set",
            Self::UnexpectedExecutableFields => "a non-executable plan contains executable target, evidence or parameter fields",
        })
    }
}
impl std::error::Error for IntentPlanIssueV1 {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn observation_commitment(
    observation: &SourceObservationV1,
) -> Result<String, IntentPlanIssueV1> {
    let payload = SourceObservationCommitmentV1 {
        schema_version: &observation.schema_version,
        source_artifact_sha256: &observation.source_artifact_sha256,
        source_lineage_sha256: &observation.source_lineage_sha256,
        observation_evidence_receipt_sha256: &observation.observation_evidence_receipt_sha256,
        duration_ticks: observation.duration_ticks,
        events: &observation.events,
        anchors: &observation.anchors,
        occurrences: &observation.occurrences,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|_| IntentPlanIssueV1::ObservationDigestMismatch)?;
    Ok(domain_sha256(OBSERVATION_DOMAIN_V1, &bytes))
}

fn validate_sorted_unique<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    let mut previous: Option<&str> = None;
    for value in values {
        if value.is_empty() || previous.is_some_and(|prior| prior >= value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

pub fn validate_source_observation_v1(
    observation: &SourceObservationV1,
) -> Result<(), IntentPlanIssueV1> {
    if observation.schema_version != SOURCE_OBSERVATION_SCHEMA_V1 {
        return Err(IntentPlanIssueV1::InvalidObservationSchema);
    }
    if observation.duration_ticks == 0 {
        return Err(IntentPlanIssueV1::InvalidDuration);
    }
    if !valid_sha256(&observation.source_artifact_sha256)
        || !valid_sha256(&observation.source_lineage_sha256)
        || !valid_sha256(&observation.observation_evidence_receipt_sha256)
        || !valid_sha256(&observation.observation_sha256)
    {
        return Err(IntentPlanIssueV1::InvalidDigest);
    }
    if observation.events.is_empty() || observation.events.len() > MAX_SOURCE_EVENTS_V1 {
        return Err(IntentPlanIssueV1::InvalidEvent);
    }

    let mut previous_event: Option<&str> = None;
    let mut event_ids = BTreeSet::new();
    for event in &observation.events {
        if event.event_id.is_empty() || event.part_id.is_empty() {
            return Err(IntentPlanIssueV1::EmptyIdentity);
        }
        if previous_event.is_some_and(|prior| prior > event.event_id.as_str()) {
            return Err(IntentPlanIssueV1::NonCanonicalEventOrder);
        }
        if !event_ids.insert(event.event_id.as_str()) {
            return Err(IntentPlanIssueV1::DuplicateEventId);
        }
        let end = event.onset_ticks.checked_add(event.duration_ticks)
            .ok_or(IntentPlanIssueV1::InvalidEvent)?;
        if event.duration_ticks == 0 || end > observation.duration_ticks {
            return Err(IntentPlanIssueV1::InvalidEvent);
        }
        previous_event = Some(&event.event_id);
    }

    let event_map: BTreeMap<&str, &SourceEventV1> = observation.events.iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect();

    if observation.anchors.len() > MAX_SOURCE_ANCHORS_V1 {
        return Err(IntentPlanIssueV1::InvalidAnchor);
    }
    let mut previous_anchor: Option<&str> = None;
    let mut anchor_ids = BTreeSet::new();
    for anchor in &observation.anchors {
        if anchor.anchor_id.is_empty() || anchor.label.trim().is_empty() {
            return Err(IntentPlanIssueV1::EmptyIdentity);
        }
        if previous_anchor.is_some_and(|prior| prior > anchor.anchor_id.as_str()) {
            return Err(IntentPlanIssueV1::NonCanonicalAnchorOrder);
        }
        if !anchor_ids.insert(anchor.anchor_id.as_str()) {
            return Err(IntentPlanIssueV1::DuplicateAnchorId);
        }
        if anchor.start_tick >= anchor.end_tick || anchor.end_tick > observation.duration_ticks {
            return Err(IntentPlanIssueV1::InvalidAnchor);
        }
        previous_anchor = Some(&anchor.anchor_id);
    }

    if observation.occurrences.len() > MAX_SOURCE_OCCURRENCES_V1 {
        return Err(IntentPlanIssueV1::InvalidOccurrence);
    }
    let mut previous_occurrence: Option<&str> = None;
    let mut occurrence_ids = BTreeSet::new();
    for occurrence in &observation.occurrences {
        if occurrence.occurrence_id.is_empty() || occurrence.semantic_class_id.is_empty() {
            return Err(IntentPlanIssueV1::EmptyIdentity);
        }
        if previous_occurrence.is_some_and(|prior| prior > occurrence.occurrence_id.as_str()) {
            return Err(IntentPlanIssueV1::NonCanonicalOccurrenceOrder);
        }
        if !occurrence_ids.insert(occurrence.occurrence_id.as_str()) {
            return Err(IntentPlanIssueV1::DuplicateOccurrenceId);
        }
        if occurrence.start_tick >= occurrence.end_tick
            || occurrence.end_tick > observation.duration_ticks
            || occurrence.event_ids.is_empty()
        {
            return Err(IntentPlanIssueV1::InvalidOccurrence);
        }
        if !validate_sorted_unique(occurrence.event_ids.iter().map(String::as_str))
            || !validate_sorted_unique(occurrence.anchor_ids.iter().map(String::as_str))
        {
            return Err(IntentPlanIssueV1::NonCanonicalReferenceOrder);
        }
        for event_id in &occurrence.event_ids {
            let event = event_map.get(event_id.as_str())
                .ok_or(IntentPlanIssueV1::UnknownEventReference)?;
            let event_end = event.onset_ticks.checked_add(event.duration_ticks)
                .ok_or(IntentPlanIssueV1::InvalidEvent)?;
            if event.onset_ticks < occurrence.start_tick || event_end > occurrence.end_tick {
                return Err(IntentPlanIssueV1::EventOutsideOccurrenceWindow);
            }
        }
        for anchor_id in &occurrence.anchor_ids {
            let anchor = observation.anchors.iter()
                .find(|anchor| anchor.anchor_id == *anchor_id)
                .ok_or(IntentPlanIssueV1::UnknownAnchorReference)?;
            if occurrence.start_tick < anchor.start_tick || occurrence.end_tick > anchor.end_tick {
                return Err(IntentPlanIssueV1::OccurrenceOutsideAnchor);
            }
        }
        previous_occurrence = Some(&occurrence.occurrence_id);
    }

    if observation_commitment(observation)? != observation.observation_sha256 {
        return Err(IntentPlanIssueV1::ObservationDigestMismatch);
    }
    Ok(())
}

/// Seal a frozen source observation after canonical ordering and reference
/// validation. The commitment covers every event, anchor and occurrence.
pub fn seal_source_observation_v1(
    mut observation: SourceObservationV1,
) -> Result<SourceObservationV1, IntentPlanIssueV1> {
    observation.observation_sha256 = "0".repeat(64);
    let digest = observation_commitment(&observation)?;
    observation.observation_sha256 = digest;
    validate_source_observation_v1(&observation)?;
    Ok(observation)
}

fn sorted_matching_occurrences<'a>(
    observation: &'a SourceObservationV1,
    selector: &OccurrenceSelectorV1,
) -> Result<Vec<&'a SourceOccurrenceV1>, IntentPlanIssueV1> {
    let mut matches: Vec<&SourceOccurrenceV1> = match selector {
        OccurrenceSelectorV1::UniqueMatching { semantic_class_id }
        | OccurrenceSelectorV1::NthMatching { semantic_class_id, .. }
        | OccurrenceSelectorV1::AllMatching { semantic_class_id }
        | OccurrenceSelectorV1::WithinAnchor { semantic_class_id, .. } => {
            if semantic_class_id.trim().is_empty() {
                return Err(IntentPlanIssueV1::InvalidSelector);
            }
            observation.occurrences.iter()
                .filter(|occurrence| occurrence.semantic_class_id == *semantic_class_id)
                .collect()
        }
    };

    if let OccurrenceSelectorV1::WithinAnchor { anchor_id, relation, .. } = selector {
        let anchor = observation.anchors.iter()
            .find(|anchor| anchor.anchor_id == *anchor_id)
            .ok_or(IntentPlanIssueV1::InvalidSelector)?;
        matches.retain(|occurrence| match relation {
            AnchorRelationV1::Inside => {
                occurrence.start_tick >= anchor.start_tick && occurrence.end_tick <= anchor.end_tick
            }
            AnchorRelationV1::After => occurrence.start_tick >= anchor.end_tick,
            AnchorRelationV1::Before => occurrence.end_tick <= anchor.start_tick,
        });
    }

    matches.sort_by(|left, right| {
        left.start_tick.cmp(&right.start_tick)
            .then_with(|| left.end_tick.cmp(&right.end_tick))
            .then_with(|| left.occurrence_id.cmp(&right.occurrence_id))
    });
    Ok(matches)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SelectorResolutionV1 {
    Ready(Vec<String>),
    NeedsClarification(Vec<String>),
    Abstain,
}

fn resolve_selector_v1(
    observation: &SourceObservationV1,
    selector: &OccurrenceSelectorV1,
) -> Result<SelectorResolutionV1, IntentPlanIssueV1> {
    let matches = sorted_matching_occurrences(observation, selector)?;
    let ids: Vec<String> = match selector {
        OccurrenceSelectorV1::UniqueMatching { .. } => {
            return Ok(match matches.len() {
                0 => SelectorResolutionV1::Abstain,
                1 => SelectorResolutionV1::Ready(vec![matches[0].occurrence_id.clone()]),
                _ => SelectorResolutionV1::NeedsClarification(
                    matches.iter().map(|occurrence| occurrence.occurrence_id.clone()).collect()
                ),
            });
        }
        OccurrenceSelectorV1::NthMatching { ordinal, .. } => {
            if *ordinal == 0 {
                return Err(IntentPlanIssueV1::InvalidSelector);
            }
            let Some(selected) = matches.get(usize::from(*ordinal - 1)) else {
                return Ok(SelectorResolutionV1::Abstain);
            };
            vec![selected.occurrence_id.clone()]
        }
        OccurrenceSelectorV1::AllMatching { .. }
        | OccurrenceSelectorV1::WithinAnchor { .. } => {
            if matches.is_empty() {
                return Ok(SelectorResolutionV1::Abstain);
            }
            matches.iter().map(|occurrence| occurrence.occurrence_id.clone()).collect()
        }
    };
    let mut ids = ids;
    ids.sort();
    Ok(SelectorResolutionV1::Ready(ids))
}

fn occurrence_by_id<'a>(
    observation: &'a SourceObservationV1,
    id: &str,
) -> Option<&'a SourceOccurrenceV1> {
    observation.occurrences.iter().find(|occurrence| occurrence.occurrence_id == id)
}

fn expected_event_support(
    observation: &SourceObservationV1,
    occurrence_ids: &[String],
) -> Result<Vec<String>, IntentPlanIssueV1> {
    let mut ids = BTreeSet::new();
    for occurrence_id in occurrence_ids {
        let occurrence = occurrence_by_id(observation, occurrence_id)
            .ok_or(IntentPlanIssueV1::SelectionSetMismatch)?;
        for event_id in &occurrence.event_ids {
            ids.insert(event_id.clone());
        }
    }
    Ok(ids.into_iter().collect())
}

fn expected_protected_complement(
    observation: &SourceObservationV1,
    selected_event_ids: &[String],
) -> Vec<String> {
    let selected: BTreeSet<&str> = selected_event_ids.iter().map(String::as_str).collect();
    observation.events.iter()
        .filter(|event| !selected.contains(event.event_id.as_str()))
        .map(|event| event.event_id.clone())
        .collect()
}

fn expected_windows(
    observation: &SourceObservationV1,
    occurrence_ids: &[String],
) -> Result<Vec<IntentTargetWindowV1>, IntentPlanIssueV1> {
    let mut windows = Vec::with_capacity(occurrence_ids.len());
    for occurrence_id in occurrence_ids {
        let occurrence = occurrence_by_id(observation, occurrence_id)
            .ok_or(IntentPlanIssueV1::SelectionSetMismatch)?;
        windows.push(IntentTargetWindowV1 {
            occurrence_id: occurrence.occurrence_id.clone(),
            start_tick: occurrence.start_tick,
            end_tick: occurrence.end_tick,
        });
    }
    windows.sort_by(|left, right| left.occurrence_id.cmp(&right.occurrence_id));
    Ok(windows)
}

fn validate_parameters(
    operation: MusicEditOperationV1,
    parameters: &[EditParameterV1],
) -> Result<(), IntentPlanIssueV1> {
    if !parameters.windows(2).all(|pair| pair[0].parameter_id < pair[1].parameter_id) {
        return Err(IntentPlanIssueV1::InvalidParameterOrder);
    }
    if parameters.iter().any(|parameter| parameter.parameter_id.trim().is_empty()) {
        return Err(IntentPlanIssueV1::EmptyIdentity);
    }
    let mut seen = BTreeSet::new();
    for parameter in parameters {
        if !seen.insert(parameter.parameter_id.as_str()) {
            return Err(IntentPlanIssueV1::DuplicateParameter);
        }
    }
    let find = |name: &str| parameters.iter().find(|parameter| parameter.parameter_id == name);
    let valid_scalar = |name: &str, unit: &str, min: i64, max: i64| {
        find(name).is_some_and(|parameter| {
            matches!(&parameter.value,
                EditParameterValueV1::Scalar { integer_value, unit: actual_unit }
                if actual_unit == unit && (*integer_value >= min && *integer_value <= max))
        })
    };
    let valid_identity_digest = |name: &str| {
        find(name).is_some_and(|parameter| {
            matches!(&parameter.value, EditParameterValueV1::Identity { value }
                if valid_sha256(value))
        })
    };
    let valid = match operation {
        MusicEditOperationV1::Add => {
            parameters.len() == 1 && valid_scalar("insertion_count", "count", 1, 64)
        }
        MusicEditOperationV1::Remove => parameters.is_empty(),
        MusicEditOperationV1::Replace => {
            parameters.len() == 1 && valid_identity_digest("replacement_asset_sha256")
        }
        MusicEditOperationV1::AdjustGain => {
            parameters.len() == 1 && valid_scalar("gain_millidb", "milli_db", -60_000, 24_000)
        }
        MusicEditOperationV1::Transpose => {
            parameters.len() == 1 && valid_scalar("semitone_delta", "semitones", -48, 48)
        }
    };
    if valid { Ok(()) } else { Err(IntentPlanIssueV1::InvalidOperationParameter) }
}

fn validate_non_executable_shape(plan: &MusicEditIntentPlanV1) -> Result<(), IntentPlanIssueV1> {
    if !plan.selected_occurrence_ids.is_empty()
        || !plan.selected_event_ids.is_empty()
        || !plan.protected_event_ids.is_empty()
        || !plan.target_windows.is_empty()
        || !plan.parameters.is_empty()
        || !plan.grounding_evidence.is_empty()
    {
        return Err(IntentPlanIssueV1::UnexpectedExecutableFields);
    }
    Ok(())
}

fn validate_plan_header(
    observation: &SourceObservationV1,
    plan: &MusicEditIntentPlanV1,
) -> Result<(), IntentPlanIssueV1> {
    if plan.schema_version != INTENT_PLAN_SCHEMA_V1 {
        return Err(IntentPlanIssueV1::InvalidPlanSchema);
    }
    if plan.plan_id.trim().is_empty() {
        return Err(IntentPlanIssueV1::EmptyIdentity);
    }
    if !valid_sha256(&plan.instruction_sha256) || !valid_sha256(&plan.source_observation_sha256) {
        return Err(IntentPlanIssueV1::InvalidDigest);
    }
    if plan.source_observation_sha256 != observation.observation_sha256 {
        return Err(IntentPlanIssueV1::SourceBindingMismatch);
    }
    if plan.confidence_bp > MAX_CONFIDENCE_BP_V1 {
        return Err(IntentPlanIssueV1::ConfidenceOutOfRange);
    }
    Ok(())
}

/// Verify a planner output against the exact source observation. This returns
/// success for valid clarification/abstention outputs, but those dispositions
/// remain non-executable and must not be translated into an EVAL-040 attempt.
pub fn verify_music_edit_intent_plan_v1(
    observation: &SourceObservationV1,
    plan: &MusicEditIntentPlanV1,
) -> Result<(), IntentPlanIssueV1> {
    validate_source_observation_v1(observation)?;
    validate_plan_header(observation, plan)?;

    match plan.disposition {
        PlanDispositionV1::NeedsClarification => {
            validate_non_executable_shape(plan)?;
            let selector = plan.selector.as_ref().ok_or(IntentPlanIssueV1::InvalidDispositionShape)?;
            if !matches!(selector, OccurrenceSelectorV1::UniqueMatching { .. })
                || plan.operation.is_none()
                || plan.clarification_question.as_deref().is_none_or(|question| question.trim().is_empty())
                || plan.abstention_reason.is_some()
            {
                return Err(IntentPlanIssueV1::InvalidDispositionShape);
            }
            let SelectorResolutionV1::NeedsClarification(expected_candidates) =
                resolve_selector_v1(observation, selector)?
            else {
                return Err(IntentPlanIssueV1::InvalidDispositionShape);
            };
            if plan.clarification_candidates != expected_candidates {
                return Err(IntentPlanIssueV1::ClarificationCandidateMismatch);
            }
            Ok(())
        }
        PlanDispositionV1::Abstain => {
            validate_non_executable_shape(plan)?;
            if plan.abstention_reason.as_deref().is_none_or(|reason| reason.trim().is_empty())
                || plan.clarification_question.is_some()
                || !plan.clarification_candidates.is_empty()
            {
                return Err(IntentPlanIssueV1::InvalidDispositionShape);
            }
            if let Some(selector) = &plan.selector {
                if matches!(resolve_selector_v1(observation, selector)?, SelectorResolutionV1::Ready(_)
                    | SelectorResolutionV1::NeedsClarification(_))
                {
                    return Err(IntentPlanIssueV1::InvalidDispositionShape);
                }
            }
            Ok(())
        }
        PlanDispositionV1::Ready => {
            let operation = plan.operation.ok_or(IntentPlanIssueV1::InvalidDispositionShape)?;
            let selector = plan.selector.as_ref().ok_or(IntentPlanIssueV1::InvalidDispositionShape)?;
            if plan.clarification_question.is_some()
                || !plan.clarification_candidates.is_empty()
                || plan.abstention_reason.is_some()
            {
                return Err(IntentPlanIssueV1::InvalidDispositionShape);
            }
            validate_parameters(operation, &plan.parameters)?;

            let SelectorResolutionV1::Ready(expected_occurrences) =
                resolve_selector_v1(observation, selector)?
            else {
                return Err(IntentPlanIssueV1::SelectionSetMismatch);
            };
            if plan.selected_occurrence_ids != expected_occurrences
                || plan.selected_occurrence_ids.is_empty()
                || plan.selected_occurrence_ids.len() > MAX_SELECTED_OCCURRENCES_V1
            {
                return Err(IntentPlanIssueV1::SelectionSetMismatch);
            }

            let expected_events = expected_event_support(observation, &expected_occurrences)?;
            if plan.selected_event_ids != expected_events {
                return Err(IntentPlanIssueV1::EventSupportMismatch);
            }
            if plan.protected_event_ids != expected_protected_complement(observation, &expected_events) {
                return Err(IntentPlanIssueV1::ProtectedComplementMismatch);
            }
            if plan.target_windows != expected_windows(observation, &expected_occurrences)? {
                return Err(IntentPlanIssueV1::TargetWindowMismatch);
            }

            if !plan.grounding_evidence.windows(2)
                .all(|pair| pair[0].occurrence_id < pair[1].occurrence_id)
            {
                return Err(IntentPlanIssueV1::GroundingEvidenceMismatch);
            }
            if plan.grounding_evidence.len() != expected_occurrences.len() {
                return Err(IntentPlanIssueV1::GroundingEvidenceMismatch);
            }
            for (evidence, occurrence_id) in plan.grounding_evidence.iter().zip(expected_occurrences.iter()) {
                if evidence.occurrence_id != *occurrence_id
                    || evidence.source_observation_sha256 != observation.observation_sha256
                    || !valid_sha256(&evidence.evidence_receipt_sha256)
                    || evidence.confidence_bp > MAX_CONFIDENCE_BP_V1
                {
                    return Err(IntentPlanIssueV1::GroundingEvidenceMismatch);
                }
            }
            Ok(())
        }
    }
}

pub fn intent_plan_commitment_v1(
    plan: &MusicEditIntentPlanV1,
) -> Result<String, IntentPlanIssueV1> {
    let bytes = serde_json::to_vec(plan).map_err(|_| IntentPlanIssueV1::InvalidPlanSchema)?;
    Ok(domain_sha256(PLAN_DOMAIN_V1, &bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(value: u64) -> String {
        format!("{value:064x}")
    }

    fn event(event_id: &str, part_id: &str, onset_ticks: u64, duration_ticks: u64) -> SourceEventV1 {
        SourceEventV1 {
            event_id: event_id.to_owned(),
            part_id: part_id.to_owned(),
            onset_ticks,
            duration_ticks,
        }
    }

    fn source() -> SourceObservationV1 {
        seal_source_observation_v1(SourceObservationV1 {
            schema_version: SOURCE_OBSERVATION_SCHEMA_V1.to_owned(),
            source_artifact_sha256: h(1),
            source_lineage_sha256: h(2),
            observation_evidence_receipt_sha256: h(90),
            duration_ticks: 6_000,
            events: vec![
                event("bass-note-1", "bass", 100, 300),
                event("guitar-note-1", "guitar", 1_200, 300),
                event("guitar-note-2", "guitar", 4_100, 300),
                event("snare-hit-1", "drums", 4_400, 100),
                event("vocal-note-1", "vocal", 4_300, 200),
            ],
            anchors: vec![
                TemporalAnchorV1 {
                    anchor_id: "bridge-1".to_owned(),
                    kind: TemporalAnchorKindV1::Section,
                    label: "bridge".to_owned(),
                    start_tick: 2_000,
                    end_tick: 3_000,
                },
                TemporalAnchorV1 {
                    anchor_id: "chorus-1".to_owned(),
                    kind: TemporalAnchorKindV1::Section,
                    label: "first chorus".to_owned(),
                    start_tick: 1_000,
                    end_tick: 2_000,
                },
                TemporalAnchorV1 {
                    anchor_id: "chorus-2".to_owned(),
                    kind: TemporalAnchorKindV1::Section,
                    label: "second chorus".to_owned(),
                    start_tick: 4_000,
                    end_tick: 5_000,
                },
            ],
            occurrences: vec![
                SourceOccurrenceV1 {
                    occurrence_id: "guitar-entry-1".to_owned(),
                    semantic_class_id: "guitar_entrance".to_owned(),
                    start_tick: 1_100,
                    end_tick: 1_600,
                    event_ids: vec!["guitar-note-1".to_owned()],
                    anchor_ids: vec!["chorus-1".to_owned()],
                },
                SourceOccurrenceV1 {
                    occurrence_id: "guitar-entry-2".to_owned(),
                    semantic_class_id: "guitar_entrance".to_owned(),
                    start_tick: 4_100,
                    end_tick: 4_500,
                    event_ids: vec!["guitar-note-2".to_owned()],
                    anchor_ids: vec!["chorus-2".to_owned()],
                },
                SourceOccurrenceV1 {
                    occurrence_id: "snare-hit-1".to_owned(),
                    semantic_class_id: "snare_hit".to_owned(),
                    start_tick: 4_400,
                    end_tick: 4_500,
                    event_ids: vec!["snare-hit-1".to_owned()],
                    anchor_ids: vec!["chorus-2".to_owned()],
                },
            ],
            observation_sha256: String::new(),
        }).expect("valid source observation")
    }

    fn base_plan(observation: &SourceObservationV1) -> MusicEditIntentPlanV1 {
        let selector = OccurrenceSelectorV1::NthMatching {
            semantic_class_id: "guitar_entrance".to_owned(),
            ordinal: 2,
        };
        let SelectorResolutionV1::Ready(occurrence_ids) =
            resolve_selector_v1(observation, &selector).unwrap()
        else { panic!("expected second occurrence"); };
        let event_ids = expected_event_support(observation, &occurrence_ids).unwrap();
        let parameters = vec![];
        MusicEditIntentPlanV1 {
            schema_version: INTENT_PLAN_SCHEMA_V1.to_owned(),
            plan_id: "plan-second-guitar-entry".to_owned(),
            instruction_sha256: h(3),
            source_observation_sha256: observation.observation_sha256.clone(),
            disposition: PlanDispositionV1::Ready,
            operation: Some(MusicEditOperationV1::Remove),
            selector: Some(selector),
            selected_occurrence_ids: occurrence_ids.clone(),
            selected_event_ids: event_ids.clone(),
            protected_event_ids: expected_protected_complement(observation, &event_ids),
            target_windows: expected_windows(observation, &occurrence_ids).unwrap(),
            parameters,
            grounding_evidence: occurrence_ids.iter().map(|id| OccurrenceGroundingEvidenceV1 {
                occurrence_id: id.clone(),
                source_observation_sha256: observation.observation_sha256.clone(),
                evidence_receipt_sha256: h(500),
                evidence_kind: GroundingEvidenceKindV1::EventMap,
                confidence_bp: 9_500,
            }).collect(),
            clarification_candidates: vec![],
            clarification_question: None,
            abstention_reason: None,
            confidence_bp: 9_500,
        }
    }

    fn unique_ambiguous_plan(observation: &SourceObservationV1) -> MusicEditIntentPlanV1 {
        let selector = OccurrenceSelectorV1::UniqueMatching {
            semantic_class_id: "guitar_entrance".to_owned(),
        };
        let SelectorResolutionV1::NeedsClarification(candidates) =
            resolve_selector_v1(observation, &selector).unwrap()
        else { panic!("expected ambiguity"); };
        MusicEditIntentPlanV1 {
            schema_version: INTENT_PLAN_SCHEMA_V1.to_owned(),
            plan_id: "plan-clarify-guitar-entry".to_owned(),
            instruction_sha256: h(4),
            source_observation_sha256: observation.observation_sha256.clone(),
            disposition: PlanDispositionV1::NeedsClarification,
            operation: Some(MusicEditOperationV1::Remove),
            selector: Some(selector),
            selected_occurrence_ids: vec![],
            selected_event_ids: vec![],
            protected_event_ids: vec![],
            target_windows: vec![],
            parameters: vec![],
            grounding_evidence: vec![],
            clarification_candidates: candidates,
            clarification_question: Some("Which guitar entrance should I remove: first chorus or second chorus?".to_owned()),
            abstention_reason: None,
            confidence_bp: 7_000,
        }
    }

    fn abstain_plan(observation: &SourceObservationV1) -> MusicEditIntentPlanV1 {
        MusicEditIntentPlanV1 {
            schema_version: INTENT_PLAN_SCHEMA_V1.to_owned(),
            plan_id: "plan-no-third-entry".to_owned(),
            instruction_sha256: h(5),
            source_observation_sha256: observation.observation_sha256.clone(),
            disposition: PlanDispositionV1::Abstain,
            operation: Some(MusicEditOperationV1::Remove),
            selector: Some(OccurrenceSelectorV1::NthMatching {
                semantic_class_id: "guitar_entrance".to_owned(),
                ordinal: 3,
            }),
            selected_occurrence_ids: vec![],
            selected_event_ids: vec![],
            protected_event_ids: vec![],
            target_windows: vec![],
            parameters: vec![],
            grounding_evidence: vec![],
            clarification_candidates: vec![],
            clarification_question: None,
            abstention_reason: Some("The source observation contains only two matching occurrences.".to_owned()),
            confidence_bp: 9_800,
        }
    }

    #[test]
    fn exact_second_occurrence_resolves_only_the_requested_event() {
        let observation = source();
        let plan = base_plan(&observation);
        verify_music_edit_intent_plan_v1(&observation, &plan).unwrap();
        assert_eq!(plan.selected_occurrence_ids, vec!["guitar-entry-2"]);
        assert_eq!(plan.selected_event_ids, vec!["guitar-note-2"]);
        assert!(plan.protected_event_ids.contains(&"guitar-note-1".to_owned()));
    }

    #[test]
    fn all_matching_selects_every_occurrence_and_only_their_event_support() {
        let observation = source();
        let selector = OccurrenceSelectorV1::AllMatching {
            semantic_class_id: "guitar_entrance".to_owned(),
        };
        let SelectorResolutionV1::Ready(ids) = resolve_selector_v1(&observation, &selector).unwrap()
        else { panic!("expected ready"); };
        let mut plan = base_plan(&observation);
        plan.selector = Some(selector);
        plan.selected_occurrence_ids = ids.clone();
        plan.selected_event_ids = expected_event_support(&observation, &ids).unwrap();
        plan.protected_event_ids = expected_protected_complement(&observation, &plan.selected_event_ids);
        plan.target_windows = expected_windows(&observation, &ids).unwrap();
        plan.grounding_evidence = ids.iter().map(|id| OccurrenceGroundingEvidenceV1 {
            occurrence_id: id.clone(),
            source_observation_sha256: observation.observation_sha256.clone(),
            evidence_receipt_sha256: h(501),
            evidence_kind: GroundingEvidenceKindV1::HumanAnnotation,
            confidence_bp: 10_000,
        }).collect();
        verify_music_edit_intent_plan_v1(&observation, &plan).unwrap();
    }

    #[test]
    fn ambiguous_unique_selector_requires_a_non_executable_clarification() {
        let observation = source();
        let plan = unique_ambiguous_plan(&observation);
        verify_music_edit_intent_plan_v1(&observation, &plan).unwrap();
        assert_ne!(plan.disposition, PlanDispositionV1::Ready);
        assert!(plan.selected_event_ids.is_empty());
        assert!(plan.parameters.is_empty());
    }

    #[test]
    fn ambiguous_selector_cannot_be_silently_promoted_to_ready() {
        let observation = source();
        let mut plan = unique_ambiguous_plan(&observation);
        plan.disposition = PlanDispositionV1::Ready;
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::SelectionSetMismatch)
        );
    }

    #[test]
    fn missing_nth_occurrence_must_abstain_not_guess() {
        let observation = source();
        let plan = abstain_plan(&observation);
        verify_music_edit_intent_plan_v1(&observation, &plan).unwrap();
        let mut fabricated = plan.clone();
        fabricated.disposition = PlanDispositionV1::Ready;
        fabricated.selected_occurrence_ids = vec!["guitar-entry-2".to_owned()];
        assert!(verify_music_edit_intent_plan_v1(&observation, &fabricated).is_err());
    }

    #[test]
    fn anchor_constrained_selector_resolves_only_occurrences_inside_second_chorus() {
        let observation = source();
        let selector = OccurrenceSelectorV1::WithinAnchor {
            semantic_class_id: "guitar_entrance".to_owned(),
            anchor_id: "chorus-2".to_owned(),
            relation: AnchorRelationV1::Inside,
        };
        let SelectorResolutionV1::Ready(ids) = resolve_selector_v1(&observation, &selector).unwrap()
        else { panic!("expected ready"); };
        assert_eq!(ids, vec!["guitar-entry-2"]);
    }

    #[test]
    fn stale_source_observation_binding_rejects() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.source_observation_sha256 = h(999);
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::SourceBindingMismatch)
        );
    }

    #[test]
    fn changed_occurrence_boundaries_cannot_expand_the_target_window() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.target_windows[0].start_tick -= 1;
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::TargetWindowMismatch)
        );
    }

    #[test]
    fn protected_event_complement_cannot_omit_untargeted_events() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.protected_event_ids.retain(|id| id != "guitar-note-1");
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::ProtectedComplementMismatch)
        );
    }

    #[test]
    fn selected_event_support_must_match_grounded_occurrence_inventory() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.selected_event_ids.push("snare-hit-1".to_owned());
        plan.selected_event_ids.sort();
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::EventSupportMismatch)
        );
    }

    #[test]
    fn operation_parameter_units_are_enforced() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.operation = Some(MusicEditOperationV1::AdjustGain);
        plan.parameters = vec![EditParameterV1 {
            parameter_id: "gain_millidb".to_owned(),
            value: EditParameterValueV1::Scalar {
                integer_value: 3_000,
                unit: "linear_gain".to_owned(),
            },
        }];
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::InvalidOperationParameter)
        );
    }

    #[test]
    fn gain_parameter_requires_declared_millidecibel_unit() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.operation = Some(MusicEditOperationV1::AdjustGain);
        plan.parameters = vec![EditParameterV1 {
            parameter_id: "gain_millidb".to_owned(),
            value: EditParameterValueV1::Scalar {
                integer_value: -6_000,
                unit: "milli_db".to_owned(),
            },
        }];
        verify_music_edit_intent_plan_v1(&observation, &plan).unwrap();
    }

    #[test]
    fn replace_operation_requires_a_bound_asset_identity() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.operation = Some(MusicEditOperationV1::Replace);
        plan.parameters = vec![EditParameterV1 {
            parameter_id: "replacement_asset_sha256".to_owned(),
            value: EditParameterValueV1::Identity { value: "not-a-digest".to_owned() },
        }];
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::InvalidOperationParameter)
        );
    }

    #[test]
    fn malformed_occurrence_evidence_receipt_digest_rejects() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.grounding_evidence[0].evidence_receipt_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::GroundingEvidenceMismatch)
        );
    }

    #[test]
    fn malformed_source_observation_receipt_digest_rejects() {
        let mut observation = source();
        observation.observation_evidence_receipt_sha256 = "not-a-digest".to_owned();
        let result = seal_source_observation_v1(observation);
        assert_eq!(result, Err(IntentPlanIssueV1::InvalidDigest));
    }

    #[test]
    fn missing_or_substituted_occurrence_evidence_rejects() {
        let observation = source();
        let mut plan = base_plan(&observation);
        plan.grounding_evidence[0].source_observation_sha256 = h(888);
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::GroundingEvidenceMismatch)
        );
    }

    #[test]
    fn observation_commitment_detects_tampered_source_inventory() {
        let observation = source();
        let mut tampered = observation.clone();
        tampered.events[0].part_id = "guitar".to_owned();
        assert_eq!(
            validate_source_observation_v1(&tampered),
            Err(IntentPlanIssueV1::ObservationDigestMismatch)
        );
    }

    #[test]
    fn clarification_candidates_must_match_the_observed_ambiguous_set() {
        let observation = source();
        let mut plan = unique_ambiguous_plan(&observation);
        plan.clarification_candidates.pop();
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::ClarificationCandidateMismatch)
        );
    }

    #[test]
    fn non_executable_plan_cannot_smuggle_a_target_or_evidence() {
        let observation = source();
        let mut plan = unique_ambiguous_plan(&observation);
        plan.selected_event_ids = vec!["guitar-note-2".to_owned()];
        assert_eq!(
            verify_music_edit_intent_plan_v1(&observation, &plan),
            Err(IntentPlanIssueV1::UnexpectedExecutableFields)
        );
    }

    #[test]
    fn confidence_never_overrides_missing_grounding() {
        let observation = source();
        let mut plan = unique_ambiguous_plan(&observation);
        plan.confidence_bp = 10_000;
        plan.disposition = PlanDispositionV1::Ready;
        assert!(verify_music_edit_intent_plan_v1(&observation, &plan).is_err());
    }

    #[test]
    fn plan_commitment_is_deterministic_and_sensitive_to_scope() {
        let observation = source();
        let plan = base_plan(&observation);
        let first = intent_plan_commitment_v1(&plan).unwrap();
        let second = intent_plan_commitment_v1(&plan).unwrap();
        assert_eq!(first, second);
        let mut altered = plan;
        altered.selected_occurrence_ids = vec!["guitar-entry-1".to_owned()];
        assert_ne!(first, intent_plan_commitment_v1(&altered).unwrap());
    }

    #[test]
    fn occurrence_must_contain_every_referenced_event() {
        let mut observation = source();
        observation.occurrences[1].start_tick = 4_200;
        let result = seal_source_observation_v1(observation);
        assert_eq!(result, Err(IntentPlanIssueV1::EventOutsideOccurrenceWindow));
    }

    #[test]
    fn occurrence_must_be_contained_by_each_declared_anchor() {
        let mut observation = source();
        observation.occurrences[1].anchor_ids = vec!["chorus-1".to_owned()];
        let result = seal_source_observation_v1(observation);
        assert_eq!(result, Err(IntentPlanIssueV1::OccurrenceOutsideAnchor));
    }

    #[test]
    fn malformed_source_event_reference_is_rejected() {
        let mut observation = source();
        observation.occurrences[0].event_ids = vec!["nonexistent-event".to_owned()];
        observation.observation_sha256 = "0".repeat(64);
        let resealed = seal_source_observation_v1(observation);
        assert_eq!(resealed, Err(IntentPlanIssueV1::UnknownEventReference));
    }
}
