// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-governed protocol contract for StreamMemBench.
//!
//! This crate validates a Rust-side host protocol and sidecar receipts. It does
//! not implement a memory engine, reader model, benchmark scorer, or Python
//! MemorySystem shim. It deliberately excludes evaluator labels from candidate
//! inputs and binds every exposed evidence item to a #13973 decision-use link.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_streammembench_adapter_v1";
pub const TIMELINE_GENESIS_V1: &str = "mace_streammembench_timeline_genesis_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMemResponseStageV1 {
    Initial,
    Revision,
    Followup,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemRunBindingV1 {
    pub schema_version: String,
    pub run_id: String,
    pub benchmark_id: String,
    pub benchmark_release: String,
    pub harness_commit_sha: String,
    pub dataset_snapshot_sha256: String,
    pub dataset_split_sha256: String,
    pub language: String,
    pub system_name: String,
    pub candidate_code_sha256: String,
    pub memory_config_sha256: String,
    pub answer_model_sha256: String,
    pub generation_config_sha256: String,
    /// The independent unit for inference is the participant, not the episode.
    pub cluster_unit: String,
    pub cluster_assignment_manifest_sha256: String,
    pub binding_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamObservationInputV1 {
    pub timestamp: String,
    pub source: String,
    pub text: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

/// Candidate-visible stream record. It contains no evidence anchors, expected
/// behavior, source spans, gold supporting observations, judge output or scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSegmentInputV1 {
    pub participant_cluster_sha256: String,
    pub day_id: String,
    pub day_ordinal: u32,
    pub segment_ordinal: u32,
    pub time_range: String,
    pub text: String,
    #[serde(default)]
    pub observations: Vec<StreamObservationInputV1>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamTimelineIngestReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub participant_cluster_sha256: String,
    pub day_id: String,
    pub day_ordinal: u32,
    pub segment_ordinal: u32,
    pub ingest_sequence: u64,
    pub stream_segment_sha256: String,
    pub previous_cursor_sha256: String,
    pub cursor_sha256: String,
    pub saved_record_manifest_sha256: String,
    pub saved_record_count: u32,
    pub ingested_at_monotonic_ns: u64,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceExposureLinkV1 {
    pub ordinal: u32,
    pub context_item_sha256: String,
    pub evidence_use_receipt_sha256: String,
    /// Canonical DecisionEvidenceUseLinkV1 digest from MACE #13973.
    pub decision_evidence_use_link_sha256: String,
    pub source_lineage_receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemResponseReceiptV1 {
    pub schema_version: String,
    pub stage: StreamMemResponseStageV1,
    pub request_sha256: String,
    pub answer_sha256: String,
    /// Digest of the exact serialized model request and answer bytes.
    pub model_request_sha256: String,
    pub model_response_sha256: String,
    pub answer_model_sha256: String,
    pub generation_config_sha256: String,
    pub context_tokens: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub latency_ns: u64,
    pub evidence_exposure: Vec<EvidenceExposureLinkV1>,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemInteractionPersistenceReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub episode_id_sha256: String,
    pub initial_request_sha256: String,
    pub initial_answer_sha256: String,
    pub feedback_sha256: String,
    pub revised_answer_sha256: Option<String>,
    pub interaction_sha256: String,
    pub persistence_confirmed: bool,
    pub persisted_at_monotonic_ns: u64,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemEpisodeReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub participant_cluster_sha256: String,
    /// Opaque evaluator-side episode identity; not the benchmark evidence ID.
    pub episode_id_sha256: String,
    pub timeline_cursor_before_sha256: String,
    pub timeline_cursor_after_sha256: String,
    pub ingested_segment_count: u32,
    pub initial_request_sha256: String,
    pub initial_response: StreamMemResponseReceiptV1,
    pub feedback_sha256: String,
    pub revision_response: Option<StreamMemResponseReceiptV1>,
    pub interaction_persistence: StreamMemInteractionPersistenceReceiptV1,
    pub followup_request_sha256: String,
    pub followup_response: StreamMemResponseReceiptV1,
    /// Optional fidelity-audit digest of records saved by stream ingestion.
    pub saved_record_manifest_sha256: Option<String>,
    pub episode_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemCommandCreateV1 {
    pub binding: StreamMemRunBindingV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemCommandGenerateV1 {
    pub episode_id_sha256: String,
    pub request_text: String,
    pub request_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemCommandFeedbackV1 {
    pub episode_id_sha256: String,
    pub request_text: String,
    pub request_sha256: String,
    pub feedback_text: String,
    pub feedback_sha256: String,
    pub initial_answer_text: String,
    pub initial_answer_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMemCommandStoreInteractionV1 {
    pub episode_id_sha256: String,
    pub user_request: String,
    pub initial_answer: String,
    pub feedback_text: String,
    pub revised_answer: Option<String>,
}

/// Tagged NDJSON command family that a thin Python MemorySystem shim can map to
/// the official StreamMemBench MemorySystem interface. Expected labels and
/// evaluator-only evidence anchors have no command representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StreamMemCommandV1 {
    Create(StreamMemCommandCreateV1),
    Reset {
        participant_cluster_sha256: String,
    },
    AddStreamToMemory {
        segment: StreamSegmentInputV1,
        ingest_sequence: u64,
        previous_cursor_sha256: String,
    },
    GenerateResponse(StreamMemCommandGenerateV1),
    GenerateResponseWithFeedback(StreamMemCommandFeedbackV1),
    StoreInteraction(StreamMemCommandStoreInteractionV1),
    RetrieveMemory {
        query_text: String,
        top_k: u32,
    },
    InspectMemoryRecords,
    Diagnostics,
    SaveMemories,
    Close {
        final_cursor_sha256: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamMemRejectV1 {
    InvalidRunBinding,
    InvalidDigest,
    InvalidInput,
    ForbiddenEvaluatorLabel,
    InvalidStreamOrder,
    DuplicateStreamSegment,
    TimelineCursorMismatch,
    TimelineEmpty,
    InvalidExposureOrder,
    DuplicateEvidenceUseLink,
    InvalidResponseReceipt,
    EpisodeStageMismatch,
    InteractionNotPersisted,
    FeedbackBindingMismatch,
    TimelineBindingMismatch,
    SerializationFailure,
}

fn digest_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty() && value.len() <= maximum
}

fn digest_serialized<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, StreamMemRejectV1> {
    serde_json::to_vec(value)
        .map(|bytes| digest_bytes(&bytes))
        .map_err(|_| StreamMemRejectV1::SerializationFailure)
}

fn forbidden_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "expectedbehavior"
            | "evidencestatement"
            | "supportingobservations"
            | "sourcespan"
            | "rawindices"
            | "fidelity"
            | "initialevidenceuse"
            | "feedbackincorporation"
            | "followupreuse"
            | "judge"
            | "judgerationale"
            | "metric"
            | "metrics"
            | "metricscores"
            | "gold"
            | "goldanswer"
            | "groundtruth"
            | "correctanswer"
            | "evaluatorlabels"
            | "feedbacktype"
            | "score"
            | "scores"
    ) || normalized.starts_with("gold")
        || normalized.starts_with("expected")
        || normalized.starts_with("judge")
}

fn contains_forbidden_key(value: &Value) -> bool {
    match value {
        Value::Object(object) => object
            .iter()
            .any(|(key, child)| forbidden_key(key) || contains_forbidden_key(child)),
        Value::Array(values) => values.iter().any(contains_forbidden_key),
        _ => false,
    }
}

impl StreamMemRunBindingV1 {
    fn unsigned_digest(&self) -> Result<String, StreamMemRejectV1> {
        let mut unsigned = self.clone();
        unsigned.binding_sha256.clear();
        digest_serialized(&unsigned)
    }

    pub fn seal(mut self) -> Result<Self, StreamMemRejectV1> {
        if self.benchmark_id != "streammembench_v1"
            || !nonempty(&self.run_id, 128)
            || !nonempty(&self.benchmark_release, 64)
            || !valid_git_sha(&self.harness_commit_sha)
            || !matches!(self.language.as_str(), "en" | "zh")
            || !nonempty(&self.system_name, 128)
            || self.cluster_unit != "participant"
        {
            return Err(StreamMemRejectV1::InvalidRunBinding);
        }
        for digest in [
            self.dataset_snapshot_sha256.as_str(),
            self.dataset_split_sha256.as_str(),
            self.candidate_code_sha256.as_str(),
            self.memory_config_sha256.as_str(),
            self.answer_model_sha256.as_str(),
            self.generation_config_sha256.as_str(),
            self.cluster_assignment_manifest_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        self.schema_version = SCHEMA_VERSION_V1.to_owned();
        self.binding_sha256.clear();
        self.binding_sha256 = self.unsigned_digest()?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        self.schema_version == SCHEMA_VERSION_V1
            && self.unsigned_digest().is_ok_and(|value| value == self.binding_sha256)
    }
}

impl StreamSegmentInputV1 {
    pub fn validate(&self) -> Result<(), StreamMemRejectV1> {
        if !valid_sha256(&self.participant_cluster_sha256)
            || !nonempty(&self.day_id, 32)
            || !nonempty(&self.time_range, 64)
            || self.text.len() > 100_000
            || self.observations.len() > 10_000
        {
            return Err(StreamMemRejectV1::InvalidInput);
        }
        let encoded = serde_json::to_value(self)
            .map_err(|_| StreamMemRejectV1::SerializationFailure)?;
        if contains_forbidden_key(&encoded) {
            return Err(StreamMemRejectV1::ForbiddenEvaluatorLabel);
        }
        for observation in &self.observations {
            if !nonempty(&observation.timestamp, 64)
                || !nonempty(&observation.source, 128)
                || observation.text.len() > 20_000
            {
                return Err(StreamMemRejectV1::InvalidInput);
            }
        }
        Ok(())
    }

    pub fn payload_sha256(&self) -> Result<String, StreamMemRejectV1> {
        self.validate()?;
        digest_serialized(self)
    }
}

impl StreamTimelineIngestReceiptV1 {
    fn unsigned_digest(&self) -> Result<String, StreamMemRejectV1> {
        let mut unsigned = self.clone();
        unsigned.receipt_sha256.clear();
        digest_serialized(&unsigned)
    }

    pub fn verify_integrity(&self) -> bool {
        self.schema_version == SCHEMA_VERSION_V1
            && self.unsigned_digest().is_ok_and(|value| value == self.receipt_sha256)
    }

    pub fn seal(mut self) -> Result<Self, StreamMemRejectV1> {
        for digest in [
            self.run_binding_sha256.as_str(),
            self.participant_cluster_sha256.as_str(),
            self.stream_segment_sha256.as_str(),
            self.previous_cursor_sha256.as_str(),
            self.saved_record_manifest_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        self.schema_version = SCHEMA_VERSION_V1.to_owned();
        self.cursor_sha256.clear();
        self.cursor_sha256 = digest_serialized(&(
            "mace_streammembench_timeline_cursor_v1",
            &self.run_id,
            &self.participant_cluster_sha256,
            &self.day_id,
            self.day_ordinal,
            self.segment_ordinal,
            self.ingest_sequence,
            &self.stream_segment_sha256,
            &self.previous_cursor_sha256,
        ))?;
        self.receipt_sha256.clear();
        self.receipt_sha256 = self.unsigned_digest()?;
        Ok(self)
    }
}

pub fn timeline_genesis_cursor_sha256_v1(
    run_id: &str,
    participant_cluster_sha256: &str,
) -> Result<String, StreamMemRejectV1> {
    if !nonempty(run_id, 128) || !valid_sha256(participant_cluster_sha256) {
        return Err(StreamMemRejectV1::InvalidInput);
    }
    digest_serialized(&(
        TIMELINE_GENESIS_V1,
        run_id,
        participant_cluster_sha256,
    ))
}

pub fn validate_timeline_receipts_v1(
    run_id: &str,
    run_binding_sha256: &str,
    participant_cluster_sha256: &str,
    receipts: &[StreamTimelineIngestReceiptV1],
) -> Result<String, StreamMemRejectV1> {
    if receipts.is_empty() {
        return Err(StreamMemRejectV1::TimelineEmpty);
    }
    if !valid_sha256(run_binding_sha256) || !valid_sha256(participant_cluster_sha256) {
        return Err(StreamMemRejectV1::InvalidDigest);
    }
    let mut expected_cursor =
        timeline_genesis_cursor_sha256_v1(run_id, participant_cluster_sha256)?;
    let mut previous_day = 0_u32;
    let mut previous_segment = 0_u32;
    let mut seen_segments = BTreeSet::new();
    for (index, receipt) in receipts.iter().enumerate() {
        if !receipt.verify_integrity()
            || receipt.run_id != run_id
            || receipt.run_binding_sha256 != run_binding_sha256
            || receipt.participant_cluster_sha256 != participant_cluster_sha256
            || receipt.ingest_sequence != index as u64
        {
            return Err(StreamMemRejectV1::InvalidStreamOrder);
        }
        if receipt.previous_cursor_sha256 != expected_cursor {
            return Err(StreamMemRejectV1::TimelineCursorMismatch);
        }
        if !seen_segments.insert((receipt.day_ordinal, receipt.segment_ordinal)) {
            return Err(StreamMemRejectV1::DuplicateStreamSegment);
        }
        if index > 0
            && (receipt.day_ordinal < previous_day
                || (receipt.day_ordinal == previous_day
                    && receipt.segment_ordinal <= previous_segment))
        {
            return Err(StreamMemRejectV1::InvalidStreamOrder);
        }
        previous_day = receipt.day_ordinal;
        previous_segment = receipt.segment_ordinal;
        expected_cursor = receipt.cursor_sha256.clone();
    }
    Ok(expected_cursor)
}

impl StreamMemResponseReceiptV1 {
    fn unsigned_digest(&self) -> Result<String, StreamMemRejectV1> {
        let mut unsigned = self.clone();
        unsigned.receipt_sha256.clear();
        digest_serialized(&unsigned)
    }

    pub fn validate(&self) -> Result<(), StreamMemRejectV1> {
        for digest in [
            self.request_sha256.as_str(),
            self.answer_sha256.as_str(),
            self.model_request_sha256.as_str(),
            self.model_response_sha256.as_str(),
            self.answer_model_sha256.as_str(),
            self.generation_config_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        let mut previous_ordinal = None;
        let mut seen_links = BTreeSet::new();
        for item in &self.evidence_exposure {
            for digest in [
                item.context_item_sha256.as_str(),
                item.evidence_use_receipt_sha256.as_str(),
                item.decision_evidence_use_link_sha256.as_str(),
                item.source_lineage_receipt_sha256.as_str(),
            ] {
                if !valid_sha256(digest) {
                    return Err(StreamMemRejectV1::InvalidDigest);
                }
            }
            if previous_ordinal.is_some_and(|value| item.ordinal <= value) {
                return Err(StreamMemRejectV1::InvalidExposureOrder);
            }
            if !seen_links.insert(item.decision_evidence_use_link_sha256.as_str()) {
                return Err(StreamMemRejectV1::DuplicateEvidenceUseLink);
            }
            previous_ordinal = Some(item.ordinal);
        }
        if !self.receipt_sha256.is_empty()
            && self.unsigned_digest()? != self.receipt_sha256
        {
            return Err(StreamMemRejectV1::InvalidResponseReceipt);
        }
        Ok(())
    }

    pub fn seal(mut self) -> Result<Self, StreamMemRejectV1> {
        self.receipt_sha256.clear();
        self.validate()?;
        self.receipt_sha256 = self.unsigned_digest()?;
        Ok(self)
    }
}

impl StreamMemInteractionPersistenceReceiptV1 {
    fn unsigned_digest(&self) -> Result<String, StreamMemRejectV1> {
        let mut unsigned = self.clone();
        unsigned.receipt_sha256.clear();
        digest_serialized(&unsigned)
    }

    pub fn seal(mut self) -> Result<Self, StreamMemRejectV1> {
        for digest in [
            self.run_binding_sha256.as_str(),
            self.episode_id_sha256.as_str(),
            self.initial_request_sha256.as_str(),
            self.initial_answer_sha256.as_str(),
            self.feedback_sha256.as_str(),
            self.interaction_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        if let Some(revised) = self.revised_answer_sha256.as_deref() {
            if !valid_sha256(revised) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        self.schema_version = SCHEMA_VERSION_V1.to_owned();
        self.receipt_sha256.clear();
        self.receipt_sha256 = self.unsigned_digest()?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        self.unsigned_digest().is_ok_and(|value| value == self.receipt_sha256)
    }
}

impl StreamMemEpisodeReceiptV1 {
    fn unsigned_digest(&self) -> Result<String, StreamMemRejectV1> {
        let mut unsigned = self.clone();
        unsigned.episode_receipt_sha256.clear();
        digest_serialized(&unsigned)
    }

    pub fn validate(&self) -> Result<(), StreamMemRejectV1> {
        if self.schema_version != SCHEMA_VERSION_V1
            || !valid_sha256(&self.run_binding_sha256)
            || !valid_sha256(&self.participant_cluster_sha256)
            || !valid_sha256(&self.episode_id_sha256)
            || !valid_sha256(&self.timeline_cursor_before_sha256)
            || !valid_sha256(&self.timeline_cursor_after_sha256)
            || !valid_sha256(&self.initial_request_sha256)
            || !valid_sha256(&self.feedback_sha256)
            || !valid_sha256(&self.followup_request_sha256)
            || self.ingested_segment_count == 0
        {
            return Err(StreamMemRejectV1::InvalidInput);
        }
        self.initial_response.validate()?;
        self.followup_response.validate()?;
        if self.initial_response.stage != StreamMemResponseStageV1::Initial
            || self.followup_response.stage != StreamMemResponseStageV1::Followup
            || self.initial_response.request_sha256 != self.initial_request_sha256
            || self.followup_response.request_sha256 != self.followup_request_sha256
        {
            return Err(StreamMemRejectV1::EpisodeStageMismatch);
        }
        if let Some(revision) = &self.revision_response {
            revision.validate()?;
            if revision.stage != StreamMemResponseStageV1::Revision
                || revision.request_sha256 != self.initial_request_sha256
            {
                return Err(StreamMemRejectV1::EpisodeStageMismatch);
            }
        }
        if !self.interaction_persistence.verify_integrity()
            || !self.interaction_persistence.persistence_confirmed
        {
            return Err(StreamMemRejectV1::InteractionNotPersisted);
        }
        let interaction = &self.interaction_persistence;
        if interaction.run_id != self.run_id
            || interaction.run_binding_sha256 != self.run_binding_sha256
            || interaction.episode_id_sha256 != self.episode_id_sha256
            || interaction.initial_request_sha256 != self.initial_request_sha256
            || interaction.initial_answer_sha256 != self.initial_response.answer_sha256
            || interaction.feedback_sha256 != self.feedback_sha256
            || interaction.revised_answer_sha256
                != self.revision_response.as_ref().map(|value| value.answer_sha256.clone())
        {
            return Err(StreamMemRejectV1::FeedbackBindingMismatch);
        }
        if self.initial_response.answer_model_sha256 != self.followup_response.answer_model_sha256
            || self.initial_response.generation_config_sha256
                != self.followup_response.generation_config_sha256
        {
            return Err(StreamMemRejectV1::EpisodeStageMismatch);
        }
        if let Some(revision) = &self.revision_response {
            if revision.answer_model_sha256 != self.initial_response.answer_model_sha256
                || revision.generation_config_sha256
                    != self.initial_response.generation_config_sha256
            {
                return Err(StreamMemRejectV1::EpisodeStageMismatch);
            }
        }
        if let Some(saved) = self.saved_record_manifest_sha256.as_deref() {
            if !valid_sha256(saved) {
                return Err(StreamMemRejectV1::InvalidDigest);
            }
        }
        if !self.episode_receipt_sha256.is_empty()
            && self.unsigned_digest()? != self.episode_receipt_sha256
        {
            return Err(StreamMemRejectV1::SerializationFailure);
        }
        Ok(())
    }

    pub fn seal(mut self) -> Result<Self, StreamMemRejectV1> {
        self.episode_receipt_sha256.clear();
        self.validate()?;
        self.episode_receipt_sha256 = self.unsigned_digest()?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        self.unsigned_digest()
            .is_ok_and(|value| value == self.episode_receipt_sha256)
            && self.validate().is_ok()
    }
}

/// Recursive label-firewall validation for every candidate-visible command.
pub fn validate_candidate_command_v1(
    command: &StreamMemCommandV1,
) -> Result<(), StreamMemRejectV1> {
    let encoded =
        serde_json::to_value(command).map_err(|_| StreamMemRejectV1::SerializationFailure)?;
    if contains_forbidden_key(&encoded) {
        return Err(StreamMemRejectV1::ForbiddenEvaluatorLabel);
    }
    match command {
        StreamMemCommandV1::AddStreamToMemory { segment, .. } => segment.validate(),
        StreamMemCommandV1::GenerateResponse(input) => {
            if !valid_sha256(&input.episode_id_sha256)
                || !nonempty(&input.request_text, 20_000)
                || digest_bytes(input.request_text.as_bytes()) != input.request_sha256
            {
                return Err(StreamMemRejectV1::InvalidInput);
            }
            Ok(())
        }
        StreamMemCommandV1::GenerateResponseWithFeedback(input) => {
            if !valid_sha256(&input.episode_id_sha256)
                || !nonempty(&input.request_text, 20_000)
                || !nonempty(&input.feedback_text, 20_000)
                || !nonempty(&input.initial_answer_text, 50_000)
                || digest_bytes(input.request_text.as_bytes()) != input.request_sha256
                || digest_bytes(input.feedback_text.as_bytes()) != input.feedback_sha256
                || digest_bytes(input.initial_answer_text.as_bytes()) != input.initial_answer_sha256
            {
                return Err(StreamMemRejectV1::InvalidInput);
            }
            Ok(())
        }
        StreamMemCommandV1::StoreInteraction(input) => {
            if !valid_sha256(&input.episode_id_sha256)
                || !nonempty(&input.user_request, 20_000)
                || !nonempty(&input.initial_answer, 50_000)
                || !nonempty(&input.feedback_text, 20_000)
                || input.revised_answer.as_deref().is_some_and(|value| value.len() > 50_000)
            {
                return Err(StreamMemRejectV1::InvalidInput);
            }
            Ok(())
        }
        StreamMemCommandV1::RetrieveMemory { query_text, top_k } => {
            if !nonempty(query_text, 20_000) || *top_k == 0 || *top_k > 1_000 {
                return Err(StreamMemRejectV1::InvalidInput);
            }
            Ok(())
        }
        StreamMemCommandV1::Reset { participant_cluster_sha256 } => {
            if valid_sha256(participant_cluster_sha256) {
                Ok(())
            } else {
                Err(StreamMemRejectV1::InvalidDigest)
            }
        }
        StreamMemCommandV1::Create(input) => {
            if input.binding.verify_integrity() {
                Ok(())
            } else {
                Err(StreamMemRejectV1::InvalidRunBinding)
            }
        }
        StreamMemCommandV1::InspectMemoryRecords
        | StreamMemCommandV1::Diagnostics
        | StreamMemCommandV1::SaveMemories
        | StreamMemCommandV1::Close { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        digest_bytes(value.as_bytes())
    }

    fn binding() -> StreamMemRunBindingV1 {
        StreamMemRunBindingV1 {
            schema_version: String::new(),
            run_id: "stream-run-001".into(),
            benchmark_id: "streammembench_v1".into(),
            benchmark_release: "v1".into(),
            harness_commit_sha: "a".repeat(40),
            dataset_snapshot_sha256: digest("dataset"),
            dataset_split_sha256: digest("split"),
            language: "en".into(),
            system_name: "mace".into(),
            candidate_code_sha256: digest("candidate"),
            memory_config_sha256: digest("memory-config"),
            answer_model_sha256: digest("answer-model"),
            generation_config_sha256: digest("generation"),
            cluster_unit: "participant".into(),
            cluster_assignment_manifest_sha256: digest("participant-map"),
            binding_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    fn segment(sequence: u64, day: u32, ordinal: u32) -> StreamTimelineIngestReceiptV1 {
        StreamTimelineIngestReceiptV1 {
            schema_version: SCHEMA_VERSION_V1.into(),
            run_id: b.run_id.clone(),
            run_binding_sha256: b.binding_sha256.clone(),
            participant_cluster_sha256: digest("participant-a"),
            day_id: format!("DAY{}", day + 1),
            day_ordinal: day,
            segment_ordinal: ordinal,
            ingest_sequence: sequence,
            stream_segment_sha256: digest(&format!("segment-{sequence}")),
            previous_cursor_sha256: String::new(),
            cursor_sha256: String::new(),
            saved_record_manifest_sha256: digest(&format!("records-{sequence}")),
            saved_record_count: 2,
            ingested_at_monotonic_ns: sequence + 10,
            receipt_sha256: String::new(),
        }
    }

    fn timeline() -> (StreamMemRunBindingV1, Vec<StreamTimelineIngestReceiptV1>) {
        let b = binding();
        let participant = digest("participant-a");
        let mut cursor = timeline_genesis_cursor_sha256_v1(&b.run_id, &participant).unwrap();
        let mut receipts = Vec::new();
        for (sequence, day, ordinal) in [(0, 0, 0), (1, 0, 1), (2, 1, 0)] {
            let mut value = segment(sequence, day, ordinal);
            value.run_id = b.run_id.clone();
            value.run_binding_sha256 = b.binding_sha256.clone();
            value.participant_cluster_sha256 = participant.clone();
            value.previous_cursor_sha256 = cursor;
            value = value.seal().unwrap();
            cursor = value.cursor_sha256.clone();
            receipts.push(value);
        }
        (b, receipts)
    }

    fn response(stage: StreamMemResponseStageV1, request: &str) -> StreamMemResponseReceiptV1 {
        StreamMemResponseReceiptV1 {
            schema_version: SCHEMA_VERSION_V1.into(),
            stage,
            request_sha256: digest(request),
            answer_sha256: digest(&format!("answer-{stage:?}")),
            model_request_sha256: digest(&format!("model-request-{stage:?}")),
            model_response_sha256: digest(&format!("model-response-{stage:?}")),
            answer_model_sha256: digest("same-model"),
            generation_config_sha256: digest("same-generation"),
            context_tokens: 100,
            prompt_tokens: 90,
            completion_tokens: 10,
            latency_ns: 1000,
            evidence_exposure: vec![EvidenceExposureLinkV1 {
                ordinal: 0,
                context_item_sha256: digest("context"),
                evidence_use_receipt_sha256: digest("use"),
                decision_evidence_use_link_sha256: digest("decision-use-link"),
                source_lineage_receipt_sha256: digest("lineage"),
            }],
            receipt_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    fn episode(with_revision: bool) -> StreamMemEpisodeReceiptV1 {
        let b = binding();
        let (b, timeline) = timeline();
        let initial_request = "initial request";
        let followup_request = "follow-up request";
        let feedback_sha = digest("feedback");
        let initial = response(StreamMemResponseStageV1::Initial, initial_request);
        let revision = with_revision
            .then(|| response(StreamMemResponseStageV1::Revision, initial_request));
        let followup = response(StreamMemResponseStageV1::Followup, followup_request);
        let interaction = StreamMemInteractionPersistenceReceiptV1 {
            schema_version: SCHEMA_VERSION_V1.into(),
            run_id: b.run_id.clone(),
            run_binding_sha256: b.binding_sha256.clone(),
            episode_id_sha256: digest("episode"),
            initial_request_sha256: digest(initial_request),
            initial_answer_sha256: initial.answer_sha256.clone(),
            feedback_sha256: feedback_sha.clone(),
            revised_answer_sha256: revision.as_ref().map(|value| value.answer_sha256.clone()),
            interaction_sha256: digest("stored-interaction"),
            persistence_confirmed: true,
            persisted_at_monotonic_ns: 5000,
            receipt_sha256: String::new(),
        }
        .seal()
        .unwrap();
        StreamMemEpisodeReceiptV1 {
            schema_version: SCHEMA_VERSION_V1.into(),
            run_id: b.run_id.clone(),
            run_binding_sha256: b.binding_sha256.clone(),
            participant_cluster_sha256: digest("participant-a"),
            episode_id_sha256: digest("episode"),
            timeline_cursor_before_sha256: timeline[0].previous_cursor_sha256.clone(),
            timeline_cursor_after_sha256: timeline[2].cursor_sha256.clone(),
            ingested_segment_count: timeline.len() as u32,
            initial_request_sha256: digest(initial_request),
            initial_response: initial,
            feedback_sha256: feedback_sha,
            revision_response: revision,
            interaction_persistence: interaction,
            followup_request_sha256: digest(followup_request),
            followup_response: followup,
            saved_record_manifest_sha256: Some(digest("saved-records")),
            episode_receipt_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    #[test]
    fn run_binding_requires_pinned_release_harness_dataset_and_participant_cluster() {
        let value = binding();
        assert!(value.verify_integrity());
        let mut changed = value;
        changed.harness_commit_sha = "not-a-commit".into();
        assert!(!changed.verify_integrity());
    }

    #[test]
    fn candidate_stream_segment_preserves_official_raw_fields() {
        let value = StreamSegmentInputV1 {
            participant_cluster_sha256: digest("participant"),
            day_id: "DAY1".into(),
            day_ordinal: 0,
            segment_ordinal: 0,
            time_range: "11:10:00-11:15:00".into(),
            text: "I set up my workstation.".into(),
            observations: vec![StreamObservationInputV1 {
                timestamp: "11:12:00".into(),
                source: "transcript".into(),
                text: "The monitor is on the left.".into(),
                metadata: BTreeMap::new(),
            }],
            metadata: BTreeMap::new(),
        };
        assert!(value.validate().is_ok());
        assert!(valid_sha256(&value.payload_sha256().unwrap()));
    }

    #[test]
    fn candidate_stream_rejects_nested_evaluator_labels() {
        let mut value = StreamSegmentInputV1 {
            participant_cluster_sha256: digest("participant"),
            day_id: "DAY1".into(),
            day_ordinal: 0,
            segment_ordinal: 0,
            time_range: "11:10:00-11:15:00".into(),
            text: "raw stream".into(),
            observations: Vec::new(),
            metadata: BTreeMap::from([(
                "trace".into(),
                serde_json::json!({"expected_behavior":"gold answer"}),
            )]),
        };
        assert_eq!(value.validate(), Err(StreamMemRejectV1::ForbiddenEvaluatorLabel));
        value.metadata.clear();
        assert!(value.validate().is_ok());
    }

    #[test]
    fn timeline_receipts_require_contiguous_order_and_cursor_chain() {
        let (binding, receipts) = timeline();
        let end = validate_timeline_receipts_v1(
            &binding.run_id,
            &binding.binding_sha256,
            &digest("participant-a"),
            &receipts,
        )
        .unwrap();
        assert_eq!(end, receipts[2].cursor_sha256);
    }

    #[test]
    fn timeline_rejects_duplicate_segments_and_sequence_gaps() {
        let (binding, mut receipts) = timeline();
        receipts[1].segment_ordinal = receipts[0].segment_ordinal;
        receipts[1].previous_cursor_sha256 = receipts[0].previous_cursor_sha256.clone();
        receipts[1] = receipts[1].clone().seal().unwrap();
        assert_eq!(
            validate_timeline_receipts_v1(
                &binding.run_id,
                &binding.binding_sha256,
                &digest("participant-a"),
                &receipts,
            ),
            Err(StreamMemRejectV1::TimelineCursorMismatch)
        );
        let (binding, mut gap) = timeline();
        gap[2].ingest_sequence = 9;
        gap[2] = gap[2].clone().seal().unwrap();
        assert_eq!(
            validate_timeline_receipts_v1(
                &binding.run_id,
                &binding.binding_sha256,
                &digest("participant-a"),
                &gap,
            ),
            Err(StreamMemRejectV1::InvalidStreamOrder)
        );
    }

    #[test]
    fn timeline_rejects_cross_participant_receipts() {
        let (binding, mut receipts) = timeline();
        receipts[2].participant_cluster_sha256 = digest("other-participant");
        receipts[2] = receipts[2].clone().seal().unwrap();
        assert_eq!(
            validate_timeline_receipts_v1(
                &binding.run_id,
                &binding.binding_sha256,
                &digest("participant-a"),
                &receipts,
            ),
            Err(StreamMemRejectV1::InvalidStreamOrder)
        );
    }

    #[test]
    fn response_receipts_require_unique_canonical_decision_use_links() {
        let mut value = response(StreamMemResponseStageV1::Initial, "request");
        assert!(value.validate().is_ok());
        value.evidence_exposure.push(value.evidence_exposure[0].clone());
        assert_eq!(value.validate(), Err(StreamMemRejectV1::InvalidExposureOrder));
        let mut duplicate = response(StreamMemResponseStageV1::Initial, "request");
        duplicate.evidence_exposure[0].ordinal = 1;
        duplicate.evidence_exposure.push(duplicate.evidence_exposure[0].clone());
        assert_eq!(duplicate.validate(), Err(StreamMemRejectV1::InvalidExposureOrder));
    }

    #[test]
    fn episode_with_revision_requires_feedback_and_persisted_interaction_before_followup() {
        let value = episode(true);
        assert!(value.validate().is_ok());
        assert!(value.verify_integrity());
        let mut changed = value;
        changed.interaction_persistence.persistence_confirmed = false;
        changed.interaction_persistence.receipt_sha256.clear();
        changed.interaction_persistence = changed.interaction_persistence.seal().unwrap();
        assert_eq!(
            changed.validate(),
            Err(StreamMemRejectV1::InteractionNotPersisted)
        );
    }

    #[test]
    fn episode_without_revision_still_requires_feedback_and_interaction_persistence() {
        let value = episode(false);
        assert!(value.validate().is_ok());
        assert!(value.revision_response.is_none());
    }

    #[test]
    fn revision_must_answer_the_same_initial_request() {
        let mut value = episode(true);
        value.revision_response.as_mut().unwrap().request_sha256 = digest("different request");
        value.revision_response.as_mut().unwrap().receipt_sha256.clear();
        *value.revision_response.as_mut().unwrap() =
            value.revision_response.as_ref().unwrap().clone().seal().unwrap();
        assert_eq!(value.validate(), Err(StreamMemRejectV1::EpisodeStageMismatch));
    }

    #[test]
    fn interaction_receipt_cannot_bind_another_initial_answer_or_feedback() {
        let mut value = episode(false);
        value.interaction_persistence.feedback_sha256 = digest("different feedback");
        value.interaction_persistence.receipt_sha256.clear();
        value.interaction_persistence = value.interaction_persistence.clone().seal().unwrap();
        assert_eq!(value.validate(), Err(StreamMemRejectV1::FeedbackBindingMismatch));
    }

    #[test]
    fn all_response_stages_share_model_and_generation_configuration() {
        let mut value = episode(true);
        value.followup_response.generation_config_sha256 = digest("different-generation");
        value.followup_response.receipt_sha256.clear();
        value.followup_response = value.followup_response.clone().seal().unwrap();
        assert_eq!(value.validate(), Err(StreamMemRejectV1::EpisodeStageMismatch));
    }

    #[test]
    fn command_contract_excludes_metric_names_and_evaluator_labels() {
        let command = StreamMemCommandV1::AddStreamToMemory {
            segment: StreamSegmentInputV1 {
                participant_cluster_sha256: digest("participant"),
                day_id: "DAY1".into(),
                day_ordinal: 0,
                segment_ordinal: 0,
                time_range: "11:10:00-11:15:00".into(),
                text: "stream text".into(),
                observations: Vec::new(),
                metadata: BTreeMap::new(),
            },
            ingest_sequence: 0,
            previous_cursor_sha256: digest("genesis"),
        };
        let encoded = serde_json::to_value(command).unwrap();
        let object = encoded.as_object().unwrap();
        assert!(object.contains_key("op"));
        assert!(!object.contains_key("expected_behavior"));
        assert!(!object.contains_key("evidence_statement"));
        assert!(!object.contains_key("feedback_incorporation"));
    }

    #[test]
    fn response_request_digest_must_match_exact_text() {
        let command = StreamMemCommandV1::GenerateResponse(StreamMemCommandGenerateV1 {
            episode_id_sha256: digest("episode"),
            request_text: "actual request".into(),
            request_sha256: digest("different request"),
        });
        assert_eq!(
            validate_candidate_command_v1(&command),
            Err(StreamMemRejectV1::InvalidInput)
        );
    }

    #[test]
    fn feedback_command_binds_feedback_and_prior_answer_bytes() {
        let valid = StreamMemCommandV1::GenerateResponseWithFeedback(StreamMemCommandFeedbackV1 {
            episode_id_sha256: digest("episode"),
            request_text: "initial request".into(),
            request_sha256: digest("initial request"),
            feedback_text: "Please correct the color.".into(),
            feedback_sha256: digest("Please correct the color."),
            initial_answer_text: "The object is red.".into(),
            initial_answer_sha256: digest("The object is red."),
        });
        assert!(validate_candidate_command_v1(&valid).is_ok());
        let mut invalid = valid;
        if let StreamMemCommandV1::GenerateResponseWithFeedback(payload) = &mut invalid {
            payload.feedback_sha256 = digest("different feedback");
        }
        assert_eq!(
            validate_candidate_command_v1(&invalid),
            Err(StreamMemRejectV1::InvalidInput)
        );
    }

    #[test]
    fn interaction_store_does_not_accept_judge_or_metric_labels() {
        let command = StreamMemCommandV1::StoreInteraction(StreamMemCommandStoreInteractionV1 {
            episode_id_sha256: digest("episode"),
            user_request: "request".into(),
            initial_answer: "initial".into(),
            feedback_text: "correct this".into(),
            revised_answer: Some("revised".into()),
        });
        assert!(validate_candidate_command_v1(&command).is_ok());
        let serialized = serde_json::to_value(&command).unwrap();
        assert!(!contains_forbidden_key(&serialized));
    }

    #[test]
    fn episode_digest_detects_receipt_mutation() {
        let value = episode(true);
        assert!(value.verify_integrity());
        let mut changed = value;
        changed.followup_response.answer_sha256 = digest("mutated answer");
        assert!(!changed.verify_integrity());
    }
}
