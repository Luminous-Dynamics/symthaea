#![forbid(unsafe_code)]

//! Label-blind adapter kernel for the official GroupMemBench retrieval contract.
//!
//! This crate is an adapter boundary, not a memory implementation, answerer,
//! judge, or benchmark reimplementation. A host injects the actual MACE runtime.
//! Candidate-side DTOs deliberately omit question IDs, question types, gold
//! answers, noise/decision-point generator flags, and judge outputs.

use std::collections::{BTreeMap, BTreeSet};

use html_escape::decode_html_entities;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROTOCOL_VERSION_V1: &str = "mace_groupmembench_adapter_v1";
pub const OFFICIAL_BENCHMARK_REPOSITORY: &str = "UCSB-NLP-Chang/GroupMemBench";
pub const PINNED_BENCHMARK_COMMIT: &str =
    "461f00c3e8a1a4a3a2ed5814b729014502c4702f";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupDomainV1 {
    Finance,
    Technology,
    Healthcare,
    Manufacturing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupMemBenchRunBindingV1 {
    pub schema_version: String,
    pub run_id: String,
    pub benchmark_repository: String,
    pub benchmark_commit_sha: String,
    pub dataset_snapshot_sha256: String,
    pub split_manifest_sha256: String,
    pub domain: GroupDomainV1,
    pub reader_model_config_sha256: String,
    pub reader_prompt_sha256: String,
    pub judge_model_config_sha256: String,
    pub judge_prompt_sha256: String,
    pub retrieval_config_sha256: String,
    pub top_k: u32,
    pub context_budget_tokens: u32,
    pub seed: u64,
    pub binding_sha256: String,
}

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, AdapterRejectV1> {
    let bytes = serde_json::to_vec(value).map_err(|_| AdapterRejectV1::SerializationFailure)?;
    Ok(digest_bytes(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl GroupMemBenchRunBindingV1 {
    /// Seal a run binding over all benchmark and comparator configuration.
    /// This digest commits to the manifest; it is not a signature or proof
    /// that the declared models were actually used.
    pub fn seal(mut self) -> Result<Self, AdapterRejectV1> {
        if !nonempty(&self.run_id)
            || self.benchmark_repository != OFFICIAL_BENCHMARK_REPOSITORY
            || !valid_git_sha(&self.benchmark_commit_sha)
            || self.top_k == 0
            || self.top_k > 256
            || self.context_budget_tokens == 0
        {
            return Err(AdapterRejectV1::InvalidRunBinding);
        }
        for digest in [
            self.dataset_snapshot_sha256.as_str(),
            self.split_manifest_sha256.as_str(),
            self.reader_model_config_sha256.as_str(),
            self.reader_prompt_sha256.as_str(),
            self.judge_model_config_sha256.as_str(),
            self.judge_prompt_sha256.as_str(),
            self.retrieval_config_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(AdapterRejectV1::InvalidDigest);
            }
        }
        self.schema_version = PROTOCOL_VERSION_V1.to_owned();
        self.binding_sha256.clear();
        self.binding_sha256 = digest_serialized(&self)?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        if self.schema_version != PROTOCOL_VERSION_V1 {
            return false;
        }
        let mut unsigned = self.clone();
        let claimed = std::mem::take(&mut unsigned.binding_sha256);
        digest_serialized(&unsigned).is_ok_and(|actual| actual == claimed)
    }
}

/// Candidate-visible message projection.
///
/// Deliberately absent: is_noise, is_decision_point, persona-generation
/// labels, gold answers, question types, and all evaluator/judge fields.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupMessageV1 {
    pub msg_node: String,
    pub content: String,
    pub author: Option<String>,
    pub role: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    pub reply_to: Option<String>,
    pub channel: String,
    pub phase_name: Option<String>,
    pub topic: Option<String>,
}

impl GroupMessageV1 {
    fn validate(&self) -> Result<(), AdapterRejectV1> {
        if !nonempty(&self.msg_node) || !nonempty(&self.channel) {
            return Err(AdapterRejectV1::InvalidMessage);
        }
        Ok(())
    }

    /// Mirrors the official BM25/dense index projection: message content only,
    /// HTML entities decoded, surrounding whitespace trimmed.
    pub fn index_text(&self) -> String {
        decode_html_entities(&self.content).trim().to_owned()
    }

    /// Mirrors the official shared RAG passage header and field ordering.
    pub fn format_retrieved_passage(&self) -> String {
        let mut tags = vec![format!("user={}", self.author.as_deref().unwrap_or("?"))];
        push_tag(&mut tags, "role", self.role.as_deref());
        push_tag(&mut tags, "channel", Some(&self.channel));
        push_tag(&mut tags, "phase_name", self.phase_name.as_deref());
        push_tag(&mut tags, "topic", self.topic.as_deref());
        push_tag(&mut tags, "timestamp", self.timestamp.as_deref());
        push_tag(&mut tags, "reply_to", self.reply_to.as_deref());
        push_tag(&mut tags, "msg_node", Some(&self.msg_node));
        format!(
            "[{}]\n{}",
            tags.join(" / "),
            decode_html_entities(&self.content).trim()
        )
    }
}

fn push_tag(tags: &mut Vec<String>, name: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        tags.push(format!("{name}={value}"));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupQueryV1 {
    pub run_id: String,
    /// Opaque per-invocation ID generated by the outer harness. It is not a
    /// dataset question ID and must not encode the task type or gold answer.
    pub query_invocation_id: String,
    pub asking_user_id: Option<String>,
    pub question_text: String,
    pub top_k: u32,
}

impl GroupQueryV1 {
    pub fn benchmark_search_query(&self) -> Result<String, AdapterRejectV1> {
        if !nonempty(&self.question_text)
            || self.asking_user_id.as_deref().is_some_and(|value| !nonempty(value))
        {
            return Err(AdapterRejectV1::InvalidQuery);
        }
        // Exact official behavior: prefix the question with asking_user_id
        // and one space when the asker is supplied.
        Ok(match self.asking_user_id.as_deref() {
            Some(asker) => format!("{asker} {}", self.question_text),
            None => self.question_text.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AdapterRequestV1 {
    Reset { run_binding: GroupMemBenchRunBindingV1 },
    Ingest { run_id: String, messages: Vec<GroupMessageV1> },
    Query { query: GroupQueryV1 },
    Close { run_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub message_count: u64,
    pub ordered_message_ids_sha256: String,
    pub candidate_projection_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievedPassageV1 {
    pub rank: u32,
    pub message_id: String,
    pub passage: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub query_invocation_id: String,
    pub search_query_sha256: String,
    pub top_k: u32,
    pub retrieved_message_ids: Vec<String>,
    pub retrieved_passages: Vec<RetrievedPassageV1>,
    pub context_projection_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AdapterResponseV1 {
    Reset { run_id: String, run_binding_sha256: String },
    Ingested { receipt: IngestReceiptV1 },
    Retrieved { receipt: QueryReceiptV1 },
    Closed { run_id: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterRejectV1 {
    InvalidJson,
    InvalidRunBinding,
    InvalidDigest,
    NoActiveRun,
    RunMismatch,
    RunNotIngested,
    AlreadyIngested,
    InvalidMessage,
    DuplicateMessageId,
    NonCanonicalMessageOrder,
    InvalidQuery,
    DuplicateQueryInvocation,
    InvalidTopK,
    UnknownRetrievedMessage,
    DuplicateRetrievedMessage,
    TooManyRetrievedMessages,
    RuntimeFailure,
    SerializationFailure,
    CloseBeforeReset,
}

impl AdapterRejectV1 {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidRunBinding => "invalid_run_binding",
            Self::InvalidDigest => "invalid_digest",
            Self::NoActiveRun => "no_active_run",
            Self::RunMismatch => "run_mismatch",
            Self::RunNotIngested => "run_not_ingested",
            Self::AlreadyIngested => "already_ingested",
            Self::InvalidMessage => "invalid_message",
            Self::DuplicateMessageId => "duplicate_message_id",
            Self::NonCanonicalMessageOrder => "non_canonical_message_order",
            Self::InvalidQuery => "invalid_query",
            Self::DuplicateQueryInvocation => "duplicate_query_invocation",
            Self::InvalidTopK => "invalid_top_k",
            Self::UnknownRetrievedMessage => "unknown_retrieved_message",
            Self::DuplicateRetrievedMessage => "duplicate_retrieved_message",
            Self::TooManyRetrievedMessages => "too_many_retrieved_messages",
            Self::RuntimeFailure => "runtime_failure",
            Self::SerializationFailure => "serialization_failure",
            Self::CloseBeforeReset => "close_before_reset",
        }
    }
}

/// Host-provided MACE memory adapter. The trait exposes only candidate-visible
/// messages and query text; it cannot receive evaluator data through this API.
pub trait GroupMemoryRuntimeV1 {
    fn reset(&mut self, binding: &GroupMemBenchRunBindingV1) -> Result<(), String>;
    fn ingest(&mut self, messages: &[GroupMessageV1]) -> Result<(), String>;
    fn retrieve(&mut self, query: &str, top_k: u32) -> Result<Vec<String>, String>;
    fn close(&mut self) -> Result<(), String>;
}

struct ActiveRun {
    binding: GroupMemBenchRunBindingV1,
    messages: Vec<GroupMessageV1>,
    by_id: BTreeMap<String, usize>,
    ingested: bool,
    seen_queries: BTreeSet<String>,
}

pub struct GroupMemBenchAdapterV1<R: GroupMemoryRuntimeV1> {
    runtime: R,
    active: Option<ActiveRun>,
}

impl<R: GroupMemoryRuntimeV1> GroupMemBenchAdapterV1<R> {
    pub fn new(runtime: R) -> Self {
        Self { runtime, active: None }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn handle(&mut self, request: AdapterRequestV1) -> Result<AdapterResponseV1, AdapterRejectV1> {
        match request {
            AdapterRequestV1::Reset { run_binding } => {
                if !run_binding.verify_integrity() {
                    return Err(AdapterRejectV1::InvalidRunBinding);
                }
                self.runtime.reset(&run_binding).map_err(|_| AdapterRejectV1::RuntimeFailure)?;
                let response = AdapterResponseV1::Reset {
                    run_id: run_binding.run_id.clone(),
                    run_binding_sha256: run_binding.binding_sha256.clone(),
                };
                self.active = Some(ActiveRun {
                    binding: run_binding,
                    messages: Vec::new(),
                    by_id: BTreeMap::new(),
                    ingested: false,
                    seen_queries: BTreeSet::new(),
                });
                Ok(response)
            }
            AdapterRequestV1::Ingest { run_id, messages } => {
                let active = self.active.as_mut().ok_or(AdapterRejectV1::NoActiveRun)?;
                if active.binding.run_id != run_id {
                    return Err(AdapterRejectV1::RunMismatch);
                }
                if active.ingested {
                    return Err(AdapterRejectV1::AlreadyIngested);
                }
                validate_messages(&messages)?;
                let mut by_id = BTreeMap::new();
                for (index, message) in messages.iter().enumerate() {
                    if by_id.insert(message.msg_node.clone(), index).is_some() {
                        return Err(AdapterRejectV1::DuplicateMessageId);
                    }
                }
                let candidate_projection_sha256 = digest_serialized(&messages)?;
                let ordered_ids: Vec<&str> = messages.iter().map(|message| message.msg_node.as_str()).collect();
                let ordered_message_ids_sha256 = digest_serialized(&ordered_ids)?;
                self.runtime.ingest(&messages).map_err(|_| AdapterRejectV1::RuntimeFailure)?;
                let receipt = IngestReceiptV1 {
                    schema_version: PROTOCOL_VERSION_V1.to_owned(),
                    run_id: run_id.clone(),
                    run_binding_sha256: active.binding.binding_sha256.clone(),
                    message_count: messages.len() as u64,
                    ordered_message_ids_sha256,
                    candidate_projection_sha256,
                };
                active.messages = messages;
                active.by_id = by_id;
                active.ingested = true;
                Ok(AdapterResponseV1::Ingested { receipt })
            }
            AdapterRequestV1::Query { query } => self.query(query),
            AdapterRequestV1::Close { run_id } => {
                let active = self.active.as_ref().ok_or(AdapterRejectV1::CloseBeforeReset)?;
                if active.binding.run_id != run_id {
                    return Err(AdapterRejectV1::RunMismatch);
                }
                self.runtime.close().map_err(|_| AdapterRejectV1::RuntimeFailure)?;
                self.active = None;
                Ok(AdapterResponseV1::Closed { run_id })
            }
        }
    }

    fn query(&mut self, query: GroupQueryV1) -> Result<AdapterResponseV1, AdapterRejectV1> {
        let active = self.active.as_mut().ok_or(AdapterRejectV1::NoActiveRun)?;
        if active.binding.run_id != query.run_id {
            return Err(AdapterRejectV1::RunMismatch);
        }
        if !active.ingested {
            return Err(AdapterRejectV1::RunNotIngested);
        }
        if !nonempty(&query.query_invocation_id) {
            return Err(AdapterRejectV1::InvalidQuery);
        }
        if query.top_k == 0 || query.top_k > active.binding.top_k || query.top_k > 256 {
            return Err(AdapterRejectV1::InvalidTopK);
        }
        if active.seen_queries.contains(&query.query_invocation_id) {
            return Err(AdapterRejectV1::DuplicateQueryInvocation);
        }
        let search_query = query.benchmark_search_query()?;
        // Consume the invocation ID before the host call. An ambiguous provider
        // failure must not permit replay of the same logical invocation.
        active.seen_queries.insert(query.query_invocation_id.clone());
        let retrieved_ids = self.runtime.retrieve(&search_query, query.top_k)
            .map_err(|_| AdapterRejectV1::RuntimeFailure)?;
        if retrieved_ids.len() > query.top_k as usize {
            return Err(AdapterRejectV1::TooManyRetrievedMessages);
        }
        let mut seen = BTreeSet::new();
        let mut passages = Vec::with_capacity(retrieved_ids.len());
        for (index, message_id) in retrieved_ids.iter().enumerate() {
            if !seen.insert(message_id.as_str()) {
                return Err(AdapterRejectV1::DuplicateRetrievedMessage);
            }
            let message_index = *active.by_id.get(message_id)
                .ok_or(AdapterRejectV1::UnknownRetrievedMessage)?;
            let message = &active.messages[message_index];
            passages.push(RetrievedPassageV1 {
                rank: (index + 1) as u32,
                message_id: message_id.clone(),
                passage: message.format_retrieved_passage(),
            });
        }
        let receipt = QueryReceiptV1 {
            schema_version: PROTOCOL_VERSION_V1.to_owned(),
            run_id: query.run_id,
            run_binding_sha256: active.binding.binding_sha256.clone(),
            query_invocation_id: query.query_invocation_id,
            search_query_sha256: digest_bytes(search_query.as_bytes()),
            top_k: query.top_k,
            retrieved_message_ids: retrieved_ids,
            context_projection_sha256: digest_serialized(&passages)?,
            retrieved_passages: passages,
        };
        Ok(AdapterResponseV1::Retrieved { receipt })
    }
}

fn validate_messages(messages: &[GroupMessageV1]) -> Result<(), AdapterRejectV1> {
    if messages.is_empty() {
        return Err(AdapterRejectV1::InvalidMessage);
    }
    let mut previous: Option<(&str, &str, &str)> = None;
    for message in messages {
        message.validate()?;
        let current = (
            message.channel.as_str(),
            message.timestamp.as_deref().unwrap_or(""),
            message.msg_node.as_str(),
        );
        if previous.is_some_and(|prior| prior > current) {
            return Err(AdapterRejectV1::NonCanonicalMessageOrder);
        }
        previous = Some(current);
    }
    Ok(())
}

/// Strict line parser: operation objects have an exact key allowlist before
/// typed deserialization. Nested run/message/query DTOs also deny unknown fields.
pub fn parse_request(line: &str) -> Result<AdapterRequestV1, AdapterRejectV1> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|_| AdapterRejectV1::InvalidJson)?;
    let object = value.as_object().ok_or(AdapterRejectV1::InvalidJson)?;
    let operation = object.get("op").and_then(serde_json::Value::as_str)
        .ok_or(AdapterRejectV1::InvalidJson)?;
    let allowed: &[&str] = match operation {
        "reset" => &["op", "run_binding"],
        "ingest" => &["op", "run_id", "messages"],
        "query" => &["op", "query"],
        "close" => &["op", "run_id"],
        _ => return Err(AdapterRejectV1::InvalidJson),
    };
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(AdapterRejectV1::InvalidJson);
    }
    serde_json::from_value(value).map_err(|_| AdapterRejectV1::InvalidJson)
}

pub fn serialize_response(response: &AdapterResponseV1) -> Result<String, AdapterRejectV1> {
    serde_json::to_string(response).map_err(|_| AdapterRejectV1::SerializationFailure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockRuntime {
        messages: Vec<GroupMessageV1>,
        last_query: Option<String>,
        forced_results: Option<Vec<String>>,
    }

    impl GroupMemoryRuntimeV1 for MockRuntime {
        fn reset(&mut self, _binding: &GroupMemBenchRunBindingV1) -> Result<(), String> {
            self.messages.clear();
            self.last_query = None;
            self.forced_results = None;
            Ok(())
        }

        fn ingest(&mut self, messages: &[GroupMessageV1]) -> Result<(), String> {
            self.messages = messages.to_vec();
            Ok(())
        }

        fn retrieve(&mut self, query: &str, top_k: u32) -> Result<Vec<String>, String> {
            self.last_query = Some(query.to_owned());
            Ok(self.forced_results.clone().unwrap_or_else(|| {
                self.messages.iter().take(top_k as usize)
                    .map(|message| message.msg_node.clone()).collect()
            }))
        }

        fn close(&mut self) -> Result<(), String> { Ok(()) }
    }

    fn sha(value: &str) -> String { digest_bytes(value.as_bytes()) }

    fn binding(run_id: &str) -> GroupMemBenchRunBindingV1 {
        GroupMemBenchRunBindingV1 {
            schema_version: String::new(),
            run_id: run_id.to_owned(),
            benchmark_repository: OFFICIAL_BENCHMARK_REPOSITORY.to_owned(),
            benchmark_commit_sha: PINNED_BENCHMARK_COMMIT.to_owned(),
            dataset_snapshot_sha256: sha("dataset"),
            split_manifest_sha256: sha("split"),
            domain: GroupDomainV1::Finance,
            reader_model_config_sha256: sha("reader-model"),
            reader_prompt_sha256: sha("reader-prompt"),
            judge_model_config_sha256: sha("judge-model"),
            judge_prompt_sha256: sha("judge-prompt"),
            retrieval_config_sha256: sha("retrieval"),
            top_k: 10,
            context_budget_tokens: 4096,
            seed: 17,
            binding_sha256: String::new(),
        }.seal().expect("valid run binding")
    }

    fn message(id: &str, channel: &str, timestamp: &str, content: &str) -> GroupMessageV1 {
        GroupMessageV1 {
            msg_node: id.to_owned(),
            content: content.to_owned(),
            author: Some("User_3".to_owned()),
            role: Some("Analyst".to_owned()),
            timestamp: Some(timestamp.to_owned()),
            reply_to: None,
            channel: channel.to_owned(),
            phase_name: Some("Planning".to_owned()),
            topic: Some("Risk".to_owned()),
        }
    }

    fn seeded_adapter(run_id: &str) -> GroupMemBenchAdapterV1<MockRuntime> {
        let mut adapter = GroupMemBenchAdapterV1::new(MockRuntime::default());
        let run = binding(run_id);
        adapter.handle(AdapterRequestV1::Reset { run_binding: run.clone() }).expect("reset");
        adapter.handle(AdapterRequestV1::Ingest {
            run_id: run.run_id,
            messages: vec![message("Msg_1", "team", "2026-01-01T00:00:00Z", "Budget is 10.")],
        }).expect("ingest");
        adapter
    }

    #[test]
    fn run_binding_is_integrity_bound() {
        let mut run = binding("run-1");
        assert!(run.verify_integrity());
        run.seed += 1;
        assert!(!run.verify_integrity());
    }

    #[test]
    fn search_query_matches_official_asking_user_prefix() {
        let query = GroupQueryV1 {
            run_id: "r".to_owned(),
            query_invocation_id: "opaque-1".to_owned(),
            asking_user_id: Some("User_7".to_owned()),
            question_text: "What is my deadline?".to_owned(),
            top_k: 10,
        };
        assert_eq!(query.benchmark_search_query().expect("valid query"),
            "User_7 What is my deadline?");
    }

    #[test]
    fn no_asker_means_question_only() {
        let query = GroupQueryV1 {
            run_id: "r".to_owned(),
            query_invocation_id: "opaque-1".to_owned(),
            asking_user_id: None,
            question_text: "What changed?".to_owned(),
            top_k: 10,
        };
        assert_eq!(query.benchmark_search_query().expect("valid query"), "What changed?");
    }

    #[test]
    fn message_projection_matches_content_only_index_and_metadata_header() {
        let value = message("Msg_1", "team", "2026-01-01T00:00:00Z", "A &amp; B");
        assert_eq!(value.index_text(), "A & B");
        assert_eq!(value.format_retrieved_passage(),
            "[user=User_3 / role=Analyst / channel=team / phase_name=Planning / topic=Risk / timestamp=2026-01-01T00:00:00Z / msg_node=Msg_1]\nA & B");
        let empty = message("Msg_2", "team", "2026-01-01T00:00:01Z", "");
        assert_eq!(empty.index_text(), "");
        assert!(validate_messages(&[empty]).is_ok());

        let mut missing_time = message("Msg_3", "team", "unused", "No timestamp.");
        missing_time.timestamp = None;
        assert!(validate_messages(&[missing_time.clone()]).is_ok());
        assert_eq!(
            missing_time.format_retrieved_passage(),
            "[user=User_3 / role=Analyst / channel=team / phase_name=Planning / topic=Risk / msg_node=Msg_3]\nNo timestamp."
        );
    }

    #[test]
    fn canonical_order_is_channel_then_timestamp_then_message_id() {
        let ordered = vec![
            message("Msg_2", "alpha", "2026-01-02T00:00:00Z", "second"),
            message("Msg_1", "alpha", "2026-01-03T00:00:00Z", "third"),
            message("Msg_3", "beta", "2026-01-01T00:00:00Z", "first"),
        ];
        assert!(validate_messages(&ordered).is_ok());
        let unordered = vec![ordered[2].clone(), ordered[0].clone()];
        assert_eq!(validate_messages(&unordered), Err(AdapterRejectV1::NonCanonicalMessageOrder));
    }

    #[test]
    fn scorer_only_fields_and_unknown_query_fields_are_rejected() {
        let raw_message = r#"{"msg_node":"Msg_1","content":"hello","author":"User_1","role":"Analyst","timestamp":"2026-01-01T00:00:00Z","reply_to":null,"channel":"team","phase_name":"Planning","topic":"Risk","is_noise":true}"#;
        assert!(serde_json::from_str::<GroupMessageV1>(raw_message).is_err());
        let raw_query = r#"{"op":"query","query":{"run_id":"r","query_invocation_id":"opaque","asking_user_id":"User_1","question_text":"hello","top_k":10,"question_type":"abstention"}}"#;
        assert_eq!(parse_request(raw_query), Err(AdapterRejectV1::InvalidJson));
    }

    #[test]
    fn unknown_top_level_fields_are_rejected() {
        let raw = r#"{"op":"close","run_id":"r","gold_answer":"leak"}"#;
        assert_eq!(parse_request(raw), Err(AdapterRejectV1::InvalidJson));
    }

    #[test]
    fn lifecycle_returns_ranked_passages_without_answer_or_judge_labels() {
        let mut adapter = seeded_adapter("run-2");
        let response = adapter.handle(AdapterRequestV1::Query {
            query: GroupQueryV1 {
                run_id: "run-2".to_owned(),
                query_invocation_id: "opaque-qa-invocation".to_owned(),
                asking_user_id: Some("User_3".to_owned()),
                question_text: "What is the budget?".to_owned(),
                top_k: 10,
            },
        }).expect("query");
        let AdapterResponseV1::Retrieved { receipt } = response else { panic!("expected retrieval result"); };
        assert_eq!(receipt.retrieved_message_ids, vec!["Msg_1"]);
        assert_eq!(receipt.retrieved_passages[0].rank, 1);
        assert!(receipt.retrieved_passages[0].passage.contains("Budget is 10."));
        assert!(adapter.runtime().last_query.as_deref().is_some_and(|query| query.starts_with("User_3 ")));
    }

    #[test]
    fn duplicate_query_invocation_is_rejected() {
        let mut adapter = seeded_adapter("run-3");
        let make_query = || GroupQueryV1 {
            run_id: "run-3".to_owned(),
            query_invocation_id: "q-1".to_owned(),
            asking_user_id: None,
            question_text: "Question".to_owned(),
            top_k: 10,
        };
        assert!(adapter.handle(AdapterRequestV1::Query { query: make_query() }).is_ok());
        assert_eq!(adapter.handle(AdapterRequestV1::Query { query: make_query() }),
            Err(AdapterRejectV1::DuplicateQueryInvocation));
    }

    #[test]
    fn unknown_or_duplicate_retrieval_ids_fail_closed() {
        let mut adapter = seeded_adapter("run-5");
        adapter.runtime.forced_results = Some(vec!["not-in-corpus".to_owned()]);
        let unknown = adapter.handle(AdapterRequestV1::Query {
            query: GroupQueryV1 {
                run_id: "run-5".to_owned(), query_invocation_id: "q-unknown".to_owned(),
                asking_user_id: None, question_text: "Question".to_owned(), top_k: 10,
            },
        });
        assert_eq!(unknown, Err(AdapterRejectV1::UnknownRetrievedMessage));

        adapter.runtime.forced_results = Some(vec!["Msg_1".to_owned(), "Msg_1".to_owned()]);
        let duplicate = adapter.handle(AdapterRequestV1::Query {
            query: GroupQueryV1 {
                run_id: "run-5".to_owned(), query_invocation_id: "q-duplicate".to_owned(),
                asking_user_id: None, question_text: "Question".to_owned(), top_k: 10,
            },
        });
        assert_eq!(duplicate, Err(AdapterRejectV1::DuplicateRetrievedMessage));
    }

    #[test]
    fn query_before_ingest_is_rejected() {
        let mut adapter = GroupMemBenchAdapterV1::new(MockRuntime::default());
        let run = binding("run-4");
        adapter.handle(AdapterRequestV1::Reset { run_binding: run.clone() }).expect("reset");
        let result = adapter.handle(AdapterRequestV1::Query {
            query: GroupQueryV1 {
                run_id: run.run_id,
                query_invocation_id: "q-1".to_owned(),
                asking_user_id: None,
                question_text: "Question".to_owned(),
                top_k: 10,
            },
        });
        assert_eq!(result, Err(AdapterRejectV1::RunNotIngested));
    }
}
