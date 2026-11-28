// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict TimlMemBench raw-native JSONL protocol boundary.
//!
//! This crate is a host-injected adapter kernel, not a memory implementation.
//! A production host must implement MaceTimlRuntimeV1 using the actual MACE
//! memory/evidence stack. The protocol boundary enforces sample isolation,
//! scorer-label exclusion, ingest receipts and evidence-qualified acceptance.
//!
//! The benchmark canonical-pool track is deliberately excluded here; it must
//! be evaluated separately from raw-native session ingestion.

use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub const PROTOCOL_VERSION_V1: &str = "timlmembench-jsonl-raw-native-v1";
pub const MACE_TRACE_METADATA_KEY_V1: &str = "mace_evidence_trace_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionPathV1 {
    AcceptAndAct,
    RetrieveMore,
    AcquireEvidence,
    Abstain,
    Escalate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceEvidenceTraceV1 {
    pub decision_path: DecisionPathV1,
    pub evidence_capsule_sha256: Option<String>,
    pub exposed_evidence_count: u64,
    pub revalidated_use_receipt_count: u64,
    pub unresolved_evidence_count: u64,
    pub verified_source_root_count: u64,
    pub dependency_closure_complete: bool,
    pub source_provenance_verified: bool,
    pub hard_gates_passed: bool,
    /// Canonical #13973 link commitments, ordered by context byte span.
    pub evidence_use_link_sha256s: Vec<String>,
}

impl MaceEvidenceTraceV1 {
    /// Fail-closed validation for the evidence-to-decision contract.
    ///
    /// This validates the host-supplied trace contract; it does not replace
    /// cryptographic verification of the underlying use receipts or capsule.
    pub fn validate(&self, answer: &str) -> Result<(), AdapterRejectV1> {
        let mut unique_links = BTreeSet::new();
        for link_sha256 in &self.evidence_use_link_sha256s {
            if !valid_sha256(link_sha256) {
                return Err(AdapterRejectV1::InvalidEvidenceDigest);
            }
            if !unique_links.insert(link_sha256.as_str()) {
                return Err(AdapterRejectV1::DuplicateEvidenceUseLink);
            }
        }
        if self.decision_path != DecisionPathV1::AcceptAndAct {
            return if answer == "<ABSTAIN>" {
                Ok(())
            } else {
                Err(AdapterRejectV1::NonAcceptingDecisionMustAbstain)
            };
        }

        let capsule = self
            .evidence_capsule_sha256
            .as_deref()
            .filter(|value| valid_sha256(value))
            .ok_or(AdapterRejectV1::AcceptWithoutEvidenceCapsule)?;
        let _ = capsule;

        if self.exposed_evidence_count == 0
            || self.revalidated_use_receipt_count < self.exposed_evidence_count
            || self.evidence_use_link_sha256s.len() as u64 != self.exposed_evidence_count
        {
            return Err(AdapterRejectV1::AcceptWithoutUseTimeRevalidation);
        }
        if self.unresolved_evidence_count > 0 {
            return Err(AdapterRejectV1::AcceptWithUnresolvedEvidence);
        }
        if !self.dependency_closure_complete {
            return Err(AdapterRejectV1::AcceptWithoutCompleteDependencyClosure);
        }
        if !self.source_provenance_verified || self.verified_source_root_count == 0 {
            return Err(AdapterRejectV1::AcceptWithoutVerifiedProvenance);
        }
        if !self.hard_gates_passed {
            return Err(AdapterRejectV1::AcceptWithHardGateFailure);
        }
        if answer.trim().is_empty() || answer == "<ABSTAIN>" {
            return Err(AdapterRejectV1::QueryResultInvalid);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceQueryResultV1 {
    pub answer: String,
    pub retrieved_memory_ids: Vec<String>,
    pub retrieved_texts: Vec<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub usage: Map<String, Value>,
    pub evidence_trace: MaceEvidenceTraceV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionIngestReceiptV1 {
    pub session_id: String,
    pub session_sha256: String,
    pub turn_count: u64,
    pub persisted: bool,
    /// Raw dialogue is not automatically trusted evidence.
    pub unverified_source_claim_count: u64,
    /// Number of source events for which the host verified receipts.
    pub verified_source_event_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimlQueryV1 {
    pub sample_id: String,
    pub language: String,
    pub task_id: String,
    pub task_type: String,
    pub case_type: Option<String>,
    pub metrics: Vec<String>,
    pub query: String,
    pub baseline: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterRejectV1 {
    InvalidJson,
    InvalidRequest,
    UnsupportedMode,
    InvalidSampleId,
    InvalidLanguage,
    NoActiveSample,
    SampleMismatch,
    LanguageMismatch,
    SessionWorldMismatch,
    OutOfOrderSession,
    SessionMustBeObject,
    SessionIdMissing,
    SessionHasNoTurns,
    InvalidTurn,
    DuplicateSessionId,
    DuplicateTurnId,
    ScorerOnlyFieldPresent,
    IngestBeforeReset,
    QueryBeforeSessionIngest,
    IngestReceiptMismatch,
    IngestNotPersisted,
    RuntimeFailure,
    Closed,
    InvalidQuery,
    QueryResultInvalid,
    DuplicateRetrievedId,
    RetrievedEvidenceCountMismatch,
    InvalidEvidenceDigest,
    DuplicateEvidenceUseLink,
    AcceptWithoutEvidenceCapsule,
    AcceptWithoutUseTimeRevalidation,
    AcceptWithUnresolvedEvidence,
    AcceptWithoutCompleteDependencyClosure,
    AcceptWithoutVerifiedProvenance,
    AcceptWithHardGateFailure,
    NonAcceptingDecisionMustAbstain,
    SerializationFailure,
    IoFailure,
}

impl AdapterRejectV1 {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedMode => "unsupported_mode",
            Self::InvalidSampleId => "invalid_sample_id",
            Self::InvalidLanguage => "invalid_language",
            Self::NoActiveSample => "no_active_sample",
            Self::SampleMismatch => "sample_mismatch",
            Self::LanguageMismatch => "language_mismatch",
            Self::SessionWorldMismatch => "session_world_mismatch",
            Self::OutOfOrderSession => "out_of_order_session",
            Self::SessionMustBeObject => "session_must_be_object",
            Self::SessionIdMissing => "session_id_missing",
            Self::SessionHasNoTurns => "session_has_no_turns",
            Self::InvalidTurn => "invalid_turn",
            Self::DuplicateSessionId => "duplicate_session_id",
            Self::DuplicateTurnId => "duplicate_turn_id",
            Self::ScorerOnlyFieldPresent => "scorer_only_field_present",
            Self::IngestBeforeReset => "ingest_before_reset",
            Self::QueryBeforeSessionIngest => "query_before_session_ingest",
            Self::IngestReceiptMismatch => "ingest_receipt_mismatch",
            Self::IngestNotPersisted => "ingest_not_persisted",
            Self::RuntimeFailure => "runtime_failure",
            Self::Closed => "closed",
            Self::InvalidQuery => "invalid_query",
            Self::QueryResultInvalid => "query_result_invalid",
            Self::DuplicateRetrievedId => "duplicate_retrieved_id",
            Self::RetrievedEvidenceCountMismatch => "retrieved_evidence_count_mismatch",
            Self::InvalidEvidenceDigest => "invalid_evidence_digest",
            Self::DuplicateEvidenceUseLink => "duplicate_evidence_use_link",
            Self::AcceptWithoutEvidenceCapsule => "accept_without_evidence_capsule",
            Self::AcceptWithoutUseTimeRevalidation => "accept_without_use_time_revalidation",
            Self::AcceptWithUnresolvedEvidence => "accept_with_unresolved_evidence",
            Self::AcceptWithoutCompleteDependencyClosure => "accept_without_complete_dependency_closure",
            Self::AcceptWithoutVerifiedProvenance => "accept_without_verified_provenance",
            Self::AcceptWithHardGateFailure => "accept_with_hard_gate_failure",
            Self::NonAcceptingDecisionMustAbstain => "non_accepting_decision_must_abstain",
            Self::SerializationFailure => "serialization_failure",
            Self::IoFailure => "io_failure",
        }
    }
}

#[derive(Debug)]
pub enum TimlRuntimeErrorV1 {
    Failed,
}

/// The host implements this interface with the actual MACE runtime. The
/// adapter owns protocol state; the host owns storage, evidence verification,
/// retrieval, decision policy and complete resource accounting.
pub trait MaceTimlRuntimeV1 {
    fn reset_sample(&mut self, sample_id: &str, language: &str) -> Result<(), TimlRuntimeErrorV1>;

    fn ingest_session(
        &mut self,
        sample_id: &str,
        language: &str,
        session: &Value,
        session_id: &str,
        session_sha256: &str,
        ingest_ordinal: u64,
    ) -> Result<SessionIngestReceiptV1, TimlRuntimeErrorV1>;

    fn query(&mut self, request: &TimlQueryV1) -> Result<MaceQueryResultV1, TimlRuntimeErrorV1>;

    fn close(&mut self) -> Result<(), TimlRuntimeErrorV1>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActiveSampleV1 {
    sample_id: String,
    language: String,
    session_ids: BTreeSet<String>,
    turn_ids: BTreeSet<String>,
    session_digests: Vec<String>,
    last_session_sort_key: Option<(String, String)>,
    total_turns: u64,
    generation: u64,
}

impl ActiveSampleV1 {
    fn ingest_manifest_sha256(&self) -> Result<String, AdapterRejectV1> {
        digest_serialized(&(
            PROTOCOL_VERSION_V1,
            self.sample_id.as_str(),
            self.language.as_str(),
            self.generation,
            self.session_digests.as_slice(),
            self.total_turns,
        ))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TimlAdapterStateV1 {
    active: Option<ActiveSampleV1>,
    generation: u64,
    closed: bool,
}

impl TimlAdapterStateV1 {
    pub fn active_sample_id(&self) -> Option<&str> {
        self.active.as_ref().map(|value| value.sample_id.as_str())
    }

    pub fn ingested_session_count(&self) -> usize {
        self.active.as_ref().map_or(0, |value| value.session_ids.len())
    }
}

#[derive(Deserialize)]
#[serde(tag = "request_type", rename_all = "snake_case", deny_unknown_fields)]
enum TimlRequestV1 {
    Reset {
        sample_id: String,
        language: String,
    },
    IngestSession {
        sample_id: String,
        language: String,
        session: Value,
    },
    Query {
        mode: String,
        baseline: String,
        language: String,
        sample_id: String,
        task_id: String,
        task_type: String,
        case_type: Option<String>,
        metrics: Vec<String>,
        query: String,
    },
    Close {},
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, AdapterRejectV1> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|_| AdapterRejectV1::SerializationFailure)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_identifier(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_len && !value.chars().any(char::is_control)
}

fn valid_language(value: &str) -> bool {
    matches!(value, "en" | "zh")
}

const SCORER_ONLY_KEYS: &[&str] = &[
    "expected_behavior",
    "expected_answer",
    "expected_answers",
    "expected_memory_ids",
    "forbidden_memory_ids",
    "forbidden_answers",
    "forbidden_answer_spans",
    "expected_answer_fragments",
    "acceptable_answers",
    "acceptable_variants",
    "unanswerable",
    "expected_current_state_ids",
    "expected_state_ids",
    "forbidden_state_ids",
    "state_id",
    "gold",
    "gold_state",
    "gold_state_id",
    "gold_state_ledger",
    "answer_key",
    "reference_answer",
    "failure_family_ids",
    "judge_rationale",
    "judge_correct",
    "judge_failure_tags",
    "forbidden_hit",
    "retrieval_correct",
    "metric_scores",
    "scorer_result",
];

fn reject_scorer_only_fields(value: &Value) -> Result<(), AdapterRejectV1> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if SCORER_ONLY_KEYS.iter().any(|forbidden| key.eq_ignore_ascii_case(forbidden)) {
                    return Err(AdapterRejectV1::ScorerOnlyFieldPresent);
                }
                reject_scorer_only_fields(child)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                reject_scorer_only_fields(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_session(session: &Value, expected_sample_id: &str) -> Result<(String, Vec<String>, u64, String, String), AdapterRejectV1> {
    let object = session.as_object().ok_or(AdapterRejectV1::SessionMustBeObject)?;
    let session_id = object.get("session_id")
        .and_then(Value::as_str)
        .filter(|value| valid_identifier(value, 256))
        .ok_or(AdapterRejectV1::SessionIdMissing)?
        .to_owned();
    let derived_sample_id = match object.get("world_id").and_then(Value::as_str) {
        Some(world_id) => world_id.to_owned(),
        None => session_id.split_once("__s").map_or_else(|| session_id.clone(), |(sample, _)| sample.to_owned()),
    };
    if derived_sample_id != expected_sample_id {
        return Err(AdapterRejectV1::SessionWorldMismatch);
    }
    let started_at = match object.get("started_at") {
        None => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Null) => "None".to_owned(),
        Some(value) => value.to_string(),
    };
    let turns = object.get("turns").and_then(Value::as_array)
        .filter(|value| !value.is_empty())
        .ok_or(AdapterRejectV1::SessionHasNoTurns)?;

    let mut turn_ids = Vec::with_capacity(turns.len());
    let mut unique = BTreeSet::new();
    for turn in turns {
        let turn_object = turn.as_object().ok_or(AdapterRejectV1::InvalidTurn)?;
        let turn_id = turn_object.get("turn_id").and_then(Value::as_str)
            .filter(|value| valid_identifier(value, 256))
            .ok_or(AdapterRejectV1::InvalidTurn)?
            .to_owned();
        let text = turn_object.get("text").and_then(Value::as_str)
            .ok_or(AdapterRejectV1::InvalidTurn)?;
        if text.trim().is_empty() {
            return Err(AdapterRejectV1::InvalidTurn);
        }
        if !unique.insert(turn_id.clone()) {
            return Err(AdapterRejectV1::DuplicateTurnId);
        }
        turn_ids.push(turn_id);
    }
    reject_scorer_only_fields(session)?;
    let session_sha256 = digest_serialized(session)?;
    Ok((session_id, turn_ids, turns.len() as u64, session_sha256, started_at))
}

fn reset_receipt_sha256(sample_id: &str, language: &str, generation: u64) -> String {
    sha256_hex(format!(
        "mace:timlmembench:sample-reset:v1\0{sample_id}\0{language}\0{generation}"
    ).as_bytes())
}

fn render_query_response(
    result: MaceQueryResultV1,
    active: &ActiveSampleV1,
) -> Result<Value, AdapterRejectV1> {
    if !valid_identifier(&result.answer, 32_000)
        || result.retrieved_memory_ids.iter().any(|id| !valid_identifier(id, 512))
        || result.retrieved_texts.iter().any(|text| text.len() > 100_000)
    {
        return Err(AdapterRejectV1::QueryResultInvalid);
    }
    if result.retrieved_memory_ids.len() != result.retrieved_texts.len() {
        return Err(AdapterRejectV1::RetrievedEvidenceCountMismatch);
    }
    if result.evidence_trace.decision_path == DecisionPathV1::AcceptAndAct
        && result.retrieved_memory_ids.is_empty()
    {
        return Err(AdapterRejectV1::AcceptWithoutEvidenceCapsule);
    }
    let mut unique = BTreeSet::new();
    if result.retrieved_memory_ids.iter().any(|id| !unique.insert(id.as_str())) {
        return Err(AdapterRejectV1::DuplicateRetrievedId);
    }
    result.evidence_trace.validate(&result.answer)?;
    let ingest_manifest_sha256 = active.ingest_manifest_sha256()?;
    let mut metadata = result.metadata;
    metadata.insert(
        MACE_TRACE_METADATA_KEY_V1.to_owned(),
        serde_json::to_value(&result.evidence_trace).map_err(|_| AdapterRejectV1::SerializationFailure)?,
    );
    metadata.insert("mace_adapter_protocol".into(), json!(PROTOCOL_VERSION_V1));
    metadata.insert("mace_ingest_manifest_sha256".into(), json!(ingest_manifest_sha256));
    metadata.insert("mace_ingested_session_count".into(), json!(active.session_ids.len()));
    metadata.insert("mace_ingested_turn_count".into(), json!(active.total_turns));
    Ok(json!({
        "answer": result.answer,
        "retrieved_memory_ids": result.retrieved_memory_ids,
        "retrieved_texts": result.retrieved_texts,
        "metadata": metadata,
        "usage": result.usage
    }))
}

/// Process one official TimlMemBench request. Invalid requests are rejected
/// before the host runtime sees them.
pub fn handle_json_line_v1<R: MaceTimlRuntimeV1>(
    runtime: &mut R,
    state: &mut TimlAdapterStateV1,
    line: &str,
) -> Result<(Value, bool), AdapterRejectV1> {
    if state.closed {
        return Err(AdapterRejectV1::Closed);
    }
    let request: TimlRequestV1 = serde_json::from_str(line).map_err(|_| AdapterRejectV1::InvalidJson)?;
    match request {
        TimlRequestV1::Reset { sample_id, language } => {
            if !valid_identifier(&sample_id, 256) { return Err(AdapterRejectV1::InvalidSampleId); }
            if !valid_language(&language) { return Err(AdapterRejectV1::InvalidLanguage); }
            if runtime.reset_sample(&sample_id, &language).is_err() {
                state.closed = true;
                let _ = runtime.close();
                return Err(AdapterRejectV1::RuntimeFailure);
            }
            state.generation = state.generation.saturating_add(1);
            state.active = Some(ActiveSampleV1 {
                sample_id: sample_id.clone(),
                language: language.clone(),
                session_ids: BTreeSet::new(),
                turn_ids: BTreeSet::new(),
                session_digests: Vec::new(),
                last_session_sort_key: None,
                total_turns: 0,
                generation: state.generation,
            });
            let reset_sha256 = reset_receipt_sha256(&sample_id, &language, state.generation);
            Ok((json!({"ok":true,"sample_id":sample_id,"reset_receipt_sha256":reset_sha256}), false))
        }
        TimlRequestV1::IngestSession { sample_id, language, session } => {
            let active = state.active.as_mut().ok_or(AdapterRejectV1::IngestBeforeReset)?;
            if sample_id != active.sample_id { return Err(AdapterRejectV1::SampleMismatch); }
            if language != active.language { return Err(AdapterRejectV1::LanguageMismatch); }
            let (session_id, turn_ids, turn_count, session_sha256, started_at) = validate_session(&session, &sample_id)?;
            let sort_key = (started_at, session_id.clone());
            if active.last_session_sort_key.as_ref().is_some_and(|previous| previous > &sort_key) {
                return Err(AdapterRejectV1::OutOfOrderSession);
            }
            if active.session_ids.contains(&session_id) { return Err(AdapterRejectV1::DuplicateSessionId); }
            if turn_ids.iter().any(|id| active.turn_ids.contains(id)) {
                return Err(AdapterRejectV1::DuplicateTurnId);
            }
            let receipt = match runtime.ingest_session(
                &sample_id, &language, &session, &session_id, &session_sha256,
                active.session_digests.len() as u64,
            ) {
                Ok(receipt) => receipt,
                Err(_) => {
                    state.closed = true;
                    let _ = runtime.close();
                    return Err(AdapterRejectV1::RuntimeFailure);
                }
            };
            if receipt.session_id != session_id
                || receipt.session_sha256 != session_sha256
                || receipt.turn_count != turn_count
            {
                state.closed = true;
                let _ = runtime.close();
                return Err(AdapterRejectV1::IngestReceiptMismatch);
            }
            if !receipt.persisted {
                state.closed = true;
                let _ = runtime.close();
                return Err(AdapterRejectV1::IngestNotPersisted);
            }
            active.session_ids.insert(session_id.clone());
            active.turn_ids.extend(turn_ids);
            active.session_digests.push(session_sha256.clone());
            active.last_session_sort_key = Some(sort_key);
            active.total_turns = active.total_turns.saturating_add(turn_count);
            Ok((json!({
                "ok":true,
                "session_id":session_id,
                "session_sha256":session_sha256,
                "turn_count":turn_count,
                "ingest_ordinal":active.session_digests.len()-1,
                "unverified_source_claim_count":receipt.unverified_source_claim_count,
                "verified_source_event_count":receipt.verified_source_event_count
            }), false))
        }
        TimlRequestV1::Query {
            mode, baseline, language, sample_id, task_id, task_type, case_type, metrics, query,
        } => {
            if mode != "raw_native_cases" { return Err(AdapterRejectV1::UnsupportedMode); }
            let active = state.active.as_ref().ok_or(AdapterRejectV1::NoActiveSample)?;
            if active.session_ids.is_empty() { return Err(AdapterRejectV1::QueryBeforeSessionIngest); }
            if sample_id != active.sample_id { return Err(AdapterRejectV1::SampleMismatch); }
            if language != active.language { return Err(AdapterRejectV1::LanguageMismatch); }
            if !valid_identifier(&task_id, 256) || !valid_identifier(&task_type, 128)
                || !valid_identifier(&baseline, 128) || query.trim().is_empty()
                || query.len() > 32_000
                || case_type.as_deref().is_some_and(|value| !valid_identifier(value, 128))
                || metrics.len() > 64
                || metrics.iter().any(|value| !valid_identifier(value, 128))
            {
                return Err(AdapterRejectV1::InvalidQuery);
            }
            let query = TimlQueryV1 {
                sample_id, language, task_id, task_type, case_type, metrics, query, baseline,
            };
            let result = match runtime.query(&query) {
                Ok(result) => result,
                Err(_) => {
                    state.closed = true;
                    let _ = runtime.close();
                    return Err(AdapterRejectV1::RuntimeFailure);
                }
            };
            Ok((render_query_response(result, active)?, false))
        }
        TimlRequestV1::Close {} => {
            runtime.close().map_err(|_| AdapterRejectV1::RuntimeFailure)?;
            state.closed = true;
            Ok((json!({"ok":true}), true))
        }
    }
}

/// Run the JSONL protocol over buffered streams. Invalid requests produce a
/// stable error code; raw dialogue and user content are never logged.
pub fn run_jsonl_v1<R, W, E>(reader: R, mut writer: W, runtime: &mut E) -> io::Result<()>
where
    R: BufRead,
    W: Write,
    E: MaceTimlRuntimeV1,
{
    let mut state = TimlAdapterStateV1::default();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let result = handle_json_line_v1(runtime, &mut state, &line);
        let (response, should_close) = match result {
            Ok(value) => (value.0, value.1),
            Err(error) => (json!({"error":error.code()}), false),
        };
        serde_json::to_writer(&mut writer, &response)
            .map_err(|_| io::Error::other("failed to serialize TimlMemBench response"))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if should_close { break; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestRuntimeV1 {
        active_sample: Option<String>,
        ingested: Vec<(String, String, u64)>,
        closed: bool,
        query_calls: usize,
        force_bad_accept: bool,
        force_good_accept: bool,
    }

    impl MaceTimlRuntimeV1 for TestRuntimeV1 {
        fn reset_sample(&mut self, sample_id: &str, _language: &str) -> Result<(), TimlRuntimeErrorV1> {
            self.active_sample = Some(sample_id.to_owned());
            self.ingested.clear();
            Ok(())
        }

        fn ingest_session(
            &mut self, sample_id: &str, _language: &str, _session: &Value,
            session_id: &str, session_sha256: &str, ordinal: u64,
        ) -> Result<SessionIngestReceiptV1, TimlRuntimeErrorV1> {
            if self.active_sample.as_deref() != Some(sample_id) { return Err(TimlRuntimeErrorV1::Failed); }
            self.ingested.push((session_id.to_owned(), session_sha256.to_owned(), ordinal));
            Ok(SessionIngestReceiptV1 {
                session_id: session_id.to_owned(), session_sha256: session_sha256.to_owned(),
                turn_count: 2, persisted: true, unverified_source_claim_count: 2,
                verified_source_event_count: 0,
            })
        }

        fn query(&mut self, _request: &TimlQueryV1) -> Result<MaceQueryResultV1, TimlRuntimeErrorV1> {
            self.query_calls += 1;
            let evidence_trace = if self.force_good_accept {
                MaceEvidenceTraceV1 {
                    decision_path: DecisionPathV1::AcceptAndAct,
                    evidence_capsule_sha256: Some("a".repeat(64)),
                    exposed_evidence_count: 1,
                    revalidated_use_receipt_count: 1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],
                    unresolved_evidence_count: 0,
                    verified_source_root_count: 1,
                    dependency_closure_complete: true,
                    source_provenance_verified: true,
                    hard_gates_passed: true,
                }
            } else if self.force_bad_accept {
                MaceEvidenceTraceV1 {
                    decision_path: DecisionPathV1::AcceptAndAct,
                    evidence_capsule_sha256: None,
                    exposed_evidence_count: 1,
                    revalidated_use_receipt_count: 0,
            evidence_use_link_sha256s: Vec::new(),
                    unresolved_evidence_count: 0,
                    verified_source_root_count: 0,
                    dependency_closure_complete: false,
                    source_provenance_verified: false,
                    hard_gates_passed: true,
                }
            } else {
                MaceEvidenceTraceV1 {
                    decision_path: DecisionPathV1::Abstain,
                    evidence_capsule_sha256: None,
                    exposed_evidence_count: 0,
                    revalidated_use_receipt_count: 0,
            evidence_use_link_sha256s: Vec::new(),
                    unresolved_evidence_count: 1,
                    verified_source_root_count: 0,
                    dependency_closure_complete: false,
                    source_provenance_verified: false,
                    hard_gates_passed: true,
                }
            };
            Ok(MaceQueryResultV1 {
                answer: if self.force_good_accept { "The latest plan is active.".into() } else { "<ABSTAIN>".into() },
                retrieved_memory_ids: if self.force_good_accept { vec!["memory-1".into()] } else { Vec::new() },
                retrieved_texts: if self.force_good_accept { vec!["The latest plan is active.".into()] } else { Vec::new() },
                metadata: Map::new(),
                usage: Map::new(),
                evidence_trace,
            })
        }

        fn close(&mut self) -> Result<(), TimlRuntimeErrorV1> {
            self.closed = true;
            Ok(())
        }
    }

    fn session(id: &str) -> Value {
        json!({
            "session_id":id,
            "world_id":if id.ends_with("b") { "sample-b" } else { "sample-a" },
            "started_at":if id.ends_with("b") { "2026-01-02T00:00:00Z" } else { "2026-01-01T00:00:00Z" },
            "surface":"private_chat",
            "turns":[
                {"turn_id":format!("{id}_t001"),"speaker_id":"A","speaker_role":"user","timestamp":"2026-01-01T00:00:00Z","text":"Please remember the new plan."},
                {"turn_id":format!("{id}_t002"),"speaker_id":"B","speaker_role":"assistant","timestamp":"2026-01-01T00:01:00Z","text":"I will keep that context."}
            ]
        })
    }

    fn reset_line(sample_id: &str) -> String {
        json!({"request_type":"reset","sample_id":sample_id,"language":"en"}).to_string()
    }

    fn ingest_line(sample_id: &str, session: Value) -> String {
        json!({"request_type":"ingest_session","sample_id":sample_id,"language":"en","session":session}).to_string()
    }

    fn query_line(sample_id: &str) -> String {
        json!({
            "request_type":"query","mode":"raw_native_cases","baseline":"mace",
            "language":"en","sample_id":sample_id,"task_id":"case-001",
            "task_type":"governance_case","case_type":"supersede_correctness",
            "metrics":["supersede_correctness"],"query":"Which plan is current?"
        }).to_string()
    }

    #[test]
    fn official_reset_ingest_query_close_sequence_is_supported() {
        let mut runtime = TestRuntimeV1::default();
        let mut state = TimlAdapterStateV1::default();
        let (reset, close) = handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        assert!(!close);
        assert!(reset["ok"].as_bool().unwrap_or(false));
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        let (response, _) = handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")).unwrap();
        assert_eq!(response["answer"],"<ABSTAIN>");
        assert_eq!(response["metadata"]["mace_ingested_session_count"],1);
        let (closed, should_close) = handle_json_line_v1(&mut runtime,&mut state,&json!({"request_type":"close"}).to_string()).unwrap();
        assert!(should_close);
        assert!(closed["ok"].as_bool().unwrap_or(false));
        assert!(runtime.closed);
    }

    #[test]
    fn raw_native_query_cannot_supply_scorer_only_fields() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        let mut request:Value=serde_json::from_str(&query_line("sample-a")).unwrap();
        request.as_object_mut().unwrap().insert("expected_behavior".into(),json!("must revoke"));
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&request.to_string()),
            Err(AdapterRejectV1::InvalidJson)
        );
        assert_eq!(runtime.query_calls,0);
    }

    #[test]
    fn scorer_only_fields_embedded_in_raw_sessions_are_rejected() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        let mut raw=session("session-1");
        raw["gold_state_ledger"]=json!([{"state_id":"secret"}]);
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",raw)),
            Err(AdapterRejectV1::ScorerOnlyFieldPresent)
        );
        assert!(runtime.ingested.is_empty());
    }

    #[test]
    fn cross_sample_ingest_and_query_are_rejected_before_runtime_call() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-b",session("session-b"))),
            Err(AdapterRejectV1::SampleMismatch)
        );
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-b")),
            Err(AdapterRejectV1::SampleMismatch)
        );
        assert!(runtime.ingested.is_empty());
        assert_eq!(runtime.query_calls,0);
    }

    #[test]
    fn duplicate_session_and_duplicate_turn_ids_are_rejected() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))),
            Err(AdapterRejectV1::DuplicateSessionId)
        );
        let mut second=session("session-2");
        second["turns"][0]["turn_id"]=json!("session-1_t001");
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",second)),
            Err(AdapterRejectV1::DuplicateTurnId)
        );
    }

    #[test]
    fn query_requires_reset_and_at_least_one_ingested_session() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")),
            Err(AdapterRejectV1::NoActiveSample)
        );
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")),
            Err(AdapterRejectV1::QueryBeforeSessionIngest)
        );
    }

    #[test]
    fn canonical_pool_track_is_separate_and_rejected_by_raw_native_adapter() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        let mut query:Value=serde_json::from_str(&query_line("sample-a")).unwrap();
        query["mode"]=json!("canonical_qa");
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query.to_string()),
            Err(AdapterRejectV1::UnsupportedMode)
        );
        assert_eq!(runtime.query_calls,0);
    }

    #[test]
    fn accepting_without_a_revalidated_capsule_is_rejected() {
        let mut runtime=TestRuntimeV1::default();
        runtime.force_bad_accept=true;
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")),
            Err(AdapterRejectV1::AcceptWithoutEvidenceCapsule)
        );
    }

    #[test]
    fn accepting_with_unresolved_or_incomplete_provenance_is_rejected() {
        let mut trace=MaceEvidenceTraceV1 {
            decision_path:DecisionPathV1::AcceptAndAct,evidence_capsule_sha256:Some("a".repeat(64)),
            exposed_evidence_count:1,revalidated_use_receipt_count:1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],unresolved_evidence_count:1,
            verified_source_root_count:1,dependency_closure_complete:false,source_provenance_verified:true,
            hard_gates_passed:true
        };
        assert_eq!(trace.validate("answer"),Err(AdapterRejectV1::AcceptWithUnresolvedEvidence));
        trace.unresolved_evidence_count=0;
        assert_eq!(trace.validate("answer"),Err(AdapterRejectV1::AcceptWithoutCompleteDependencyClosure));
        trace.dependency_closure_complete=true;
        trace.source_provenance_verified=false;
        assert_eq!(trace.validate("answer"),Err(AdapterRejectV1::AcceptWithoutVerifiedProvenance));
    }

    #[test]
    fn hard_gate_failure_and_non_accepting_paths_require_abstention() {
        let trace=MaceEvidenceTraceV1 {
            decision_path:DecisionPathV1::AcceptAndAct,evidence_capsule_sha256:Some("a".repeat(64)),
            exposed_evidence_count:1,revalidated_use_receipt_count:1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],unresolved_evidence_count:0,
            verified_source_root_count:1,dependency_closure_complete:true,source_provenance_verified:true,
            hard_gates_passed:false
        };
        assert_eq!(trace.validate("answer"),Err(AdapterRejectV1::AcceptWithHardGateFailure));
        let non_accept=MaceEvidenceTraceV1 {
            decision_path:DecisionPathV1::RetrieveMore,evidence_capsule_sha256:None,
            exposed_evidence_count:0,revalidated_use_receipt_count:0,
            evidence_use_link_sha256s: Vec::new(),unresolved_evidence_count:1,
            verified_source_root_count:0,dependency_closure_complete:false,source_provenance_verified:false,
            hard_gates_passed:true
        };
        assert_eq!(non_accept.validate("not abstaining"),Err(AdapterRejectV1::NonAcceptingDecisionMustAbstain));
    }

    #[test]
    fn ingest_receipt_must_bind_persisted_session_bytes_and_turn_count() {
        struct BadReceiptRuntime(TestRuntimeV1);
        impl MaceTimlRuntimeV1 for BadReceiptRuntime {
            fn reset_sample(&mut self,s:&str,l:&str)->Result<(),TimlRuntimeErrorV1>{self.0.reset_sample(s,l)}
            fn ingest_session(&mut self,_:&str,_:&str,_:&Value,id:&str,sha:&str,_:u64)->Result<SessionIngestReceiptV1,TimlRuntimeErrorV1>{
                Ok(SessionIngestReceiptV1{session_id:id.into(),session_sha256:sha.into(),turn_count:99,persisted:false,unverified_source_claim_count:0,verified_source_event_count:0})
            }
            fn query(&mut self,q:&TimlQueryV1)->Result<MaceQueryResultV1,TimlRuntimeErrorV1>{self.0.query(q)}
            fn close(&mut self)->Result<(),TimlRuntimeErrorV1>{self.0.close()}
        }
        let mut runtime=BadReceiptRuntime(TestRuntimeV1::default());
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))),
            Err(AdapterRejectV1::IngestReceiptMismatch)
        );
        assert_eq!(state.ingested_session_count(),0);
        assert!(state.closed);
        assert!(runtime.0.closed);
        assert_eq!(handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")),Err(AdapterRejectV1::Closed));
    }

    #[test]
    fn reset_clears_sample_scope_and_second_sample_starts_new_manifest() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-a"))).unwrap();
        let first=state.active.as_ref().unwrap().ingest_manifest_sha256().unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-b")).unwrap();
        assert_eq!(state.active_sample_id(),Some("sample-b"));
        assert_eq!(state.ingested_session_count(),0);
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-b",session("session-b"))).unwrap();
        let second=state.active.as_ref().unwrap().ingest_manifest_sha256().unwrap();
        assert_ne!(first,second);
    }

    #[test]
    fn run_jsonl_emits_one_response_per_request_and_stops_on_close() {
        let mut runtime=TestRuntimeV1::default();
        let input=[
            reset_line("sample-a"),
            ingest_line("sample-a",session("session-1")),
            query_line("sample-a"),
            json!({"request_type":"close"}).to_string(),
            query_line("sample-a")
        ].join("\n")+"\n";
        let mut output=Vec::new();
        run_jsonl_v1(input.as_bytes(),&mut output,&mut runtime).unwrap();
        let lines=String::from_utf8(output).unwrap().lines().count();
        assert_eq!(lines,4);
        assert!(runtime.closed);
    }

    #[test]
    fn accepting_with_complete_revalidated_evidence_is_serialized() {
        let mut runtime=TestRuntimeV1::default();
        runtime.force_good_accept=true;
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))).unwrap();
        let (response,_)=handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")).unwrap();
        assert_eq!(response["answer"],"The latest plan is active.");
        assert_eq!(response["retrieved_memory_ids"][0],"memory-1");
        assert_eq!(response["metadata"][MACE_TRACE_METADATA_KEY_V1]["decision_path"],"accept_and_act");
        assert_eq!(response["metadata"]["mace_ingested_session_count"],1);
    }

    #[test]
    fn accepting_without_retrieved_ids_is_rejected() {
        let trace=MaceEvidenceTraceV1 {
            decision_path:DecisionPathV1::AcceptAndAct,
            evidence_capsule_sha256:Some("a".repeat(64)),
            exposed_evidence_count:1,
            revalidated_use_receipt_count:1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],
            unresolved_evidence_count:0,
            verified_source_root_count:1,
            dependency_closure_complete:true,
            source_provenance_verified:true,
            hard_gates_passed:true
        };
        let result=MaceQueryResultV1 {
            answer:"grounded answer".into(),
            retrieved_memory_ids:Vec::new(),
            retrieved_texts:Vec::new(),
            metadata:Map::new(),
            usage:Map::new(),
            evidence_trace:trace
        };
        let active=ActiveSampleV1 {
            sample_id:"sample-a".into(),language:"en".into(),
            session_ids:BTreeSet::from(["session-1".into()]),
            turn_ids:BTreeSet::from(["turn-1".into()]),
            session_digests:vec!["b".repeat(64)],last_session_sort_key:None,
            total_turns:1,generation:1
        };
        assert_eq!(render_query_response(result,&active),Err(AdapterRejectV1::AcceptWithoutEvidenceCapsule));
    }


    #[test]
    fn sessions_must_match_world_and_official_started_at_sort_order() {
        let mut runtime=TestRuntimeV1::default();
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        let first=session("session-a");
        handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",first)).unwrap();

        let mut wrong_world=session("session-b");
        wrong_world["world_id"]=json!("sample-b");
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",wrong_world)),
            Err(AdapterRejectV1::SessionWorldMismatch)
        );

        let mut earlier=session("session-c");
        earlier["world_id"]=json!("sample-a");
        earlier["started_at"]=json!("2025-12-31T23:59:59Z");
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",earlier)),
            Err(AdapterRejectV1::OutOfOrderSession)
        );
    }


    #[test]
    fn runtime_ingest_error_fail_stops_after_unknown_mutation_outcome() {
        struct UnknownIngestRuntime(TestRuntimeV1);
        impl MaceTimlRuntimeV1 for UnknownIngestRuntime {
            fn reset_sample(&mut self,s:&str,l:&str)->Result<(),TimlRuntimeErrorV1>{self.0.reset_sample(s,l)}
            fn ingest_session(&mut self,_:&str,_:&str,_:&Value,_:&str,_:&str,_:u64)->Result<SessionIngestReceiptV1,TimlRuntimeErrorV1>{
                self.0.ingested.push(("possibly-persisted".into(),"unknown".into(),0));
                Err(TimlRuntimeErrorV1::Failed)
            }
            fn query(&mut self,q:&TimlQueryV1)->Result<MaceQueryResultV1,TimlRuntimeErrorV1>{self.0.query(q)}
            fn close(&mut self)->Result<(),TimlRuntimeErrorV1>{self.0.close()}
        }
        let mut runtime=UnknownIngestRuntime(TestRuntimeV1::default());
        let mut state=TimlAdapterStateV1::default();
        handle_json_line_v1(&mut runtime,&mut state,&reset_line("sample-a")).unwrap();
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&ingest_line("sample-a",session("session-1"))),
            Err(AdapterRejectV1::RuntimeFailure)
        );
        assert!(state.closed);
        assert!(runtime.0.closed);
        assert_eq!(
            handle_json_line_v1(&mut runtime,&mut state,&query_line("sample-a")),
            Err(AdapterRejectV1::Closed)
        );
    }


    #[test]
    fn accepted_trace_requires_one_use_receipt_per_exposed_item() {
        let trace=MaceEvidenceTraceV1{
            decision_path:DecisionPathV1::AcceptAndAct,
            evidence_capsule_sha256:Some("a".repeat(64)),
            exposed_evidence_count:2,revalidated_use_receipt_count:1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],
            unresolved_evidence_count:0,verified_source_root_count:1,
            dependency_closure_complete:true,source_provenance_verified:true,
            hard_gates_passed:true
        };
        assert_eq!(trace.validate("grounded answer"),Err(AdapterRejectV1::AcceptWithoutUseTimeRevalidation));
    }

    #[test]
    fn accepted_trace_rejects_malformed_capsule_and_empty_answer() {
        let mut trace=MaceEvidenceTraceV1{
            decision_path:DecisionPathV1::AcceptAndAct,
            evidence_capsule_sha256:Some("not-a-digest".into()),
            exposed_evidence_count:1,revalidated_use_receipt_count:1,
            evidence_use_link_sha256s: vec!["c".repeat(64)],
            unresolved_evidence_count:0,verified_source_root_count:1,
            dependency_closure_complete:true,source_provenance_verified:true,
            hard_gates_passed:true
        };
        assert_eq!(trace.validate("grounded answer"),Err(AdapterRejectV1::AcceptWithoutEvidenceCapsule));
        trace.evidence_capsule_sha256=Some("a".repeat(64));
        assert_eq!(trace.validate("   "),Err(AdapterRejectV1::QueryResultInvalid));
    }


    #[test]
    fn accepted_trace_requires_unique_canonical_per_span_links() {
        let mut trace=MaceEvidenceTraceV1{
            decision_path:DecisionPathV1::AcceptAndAct,
            evidence_capsule_sha256:Some("a".repeat(64)),
            exposed_evidence_count:1,revalidated_use_receipt_count:1,
            evidence_use_link_sha256s:Vec::new(),
            unresolved_evidence_count:0,verified_source_root_count:1,
            dependency_closure_complete:true,source_provenance_verified:true,hard_gates_passed:true
        };
        assert_eq!(trace.validate("grounded"),Err(AdapterRejectV1::AcceptWithoutUseTimeRevalidation));
        trace.evidence_use_link_sha256s=vec!["invalid".into()];
        assert_eq!(trace.validate("grounded"),Err(AdapterRejectV1::InvalidEvidenceDigest));
        trace.evidence_use_link_sha256s=vec!["c".repeat(64),"c".repeat(64)];
        trace.exposed_evidence_count=2;
        trace.revalidated_use_receipt_count=2;
        assert_eq!(trace.validate("grounded"),Err(AdapterRejectV1::DuplicateEvidenceUseLink));
    }

}
