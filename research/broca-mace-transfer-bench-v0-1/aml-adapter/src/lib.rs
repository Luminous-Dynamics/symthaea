// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provider-neutral Add/Search protocol core for a future AML adapter.
//!
//! This crate validates the public API boundary and deterministic Textual
//! chunking. It is not an HTTP server, memory database, embedding service or
//! AML submission. A hosted service must acknowledge Add only after durable
//! persistence and immediate Search visibility.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const FORMAL_TOP_K: usize = 100;
pub const TEXTUAL_MAX_MESSAGES: usize = 20;
pub const TEXTUAL_MAX_WORDS: usize = 2_000;
pub const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_SEARCH_RESPONSE_IMAGE_BYTES: usize = 30 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Track {
    Textual,
    Coding,
    Multimodal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentPart {
    Text(String),
    ImageDataUri(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub timestamp_ms: Option<i64>,
    pub content: MessageContent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddRequest {
    pub request_id: String,
    pub messages: Vec<Message>,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchRequest {
    pub query: MessageContent,
    pub options: Option<Vec<String>>,
    pub user_id: String,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchItem {
    pub id: String,
    pub content: MessageContent,
    pub score: Option<f64>,
    pub created_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResponse {
    pub data: Vec<SearchItem>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AmlReject {
    EmptyRequestId,
    EmptyUserId,
    EmptySessionId,
    EmptyMessages,
    TooManyMessages,
    EmptyText,
    EmptyContentParts,
    TrackContentMismatch,
    InvalidTimestamp,
    InvalidImageDataUri,
    InvalidBase64,
    ImageTooLarge,
    SearchImageBudgetExceeded,
    EmptyOptions,
    EmptyOption,
    TooManyWords,
    EmptyCreatedAt,
    InvalidTopK,
    EmptySearchUserId,
    AddNotPersistedAndSearchable,
    AddResponseNotSuccessful,
    AddResponseEchoMismatch,
    SearchResponseTooLarge,
    EmptyMemoryId,
    DuplicateMemoryId,
    EmptyMemoryContent,
    InvalidScore,
    RequestIdPayloadConflict,
    UnknownRequestId,
    AddAlreadyCompleted,
    WordCounterVersionMissing,
    SingleMessageWordLimitExceeded,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn image_data_uri_decoded_len(value: &str) -> Result<usize, AmlReject> {
    let (header, payload) = value
        .split_once(',')
        .ok_or(AmlReject::InvalidImageDataUri)?;
    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
        return Err(AmlReject::InvalidImageDataUri);
    }
    if payload.is_empty() || payload.len() % 4 != 0 {
        return Err(AmlReject::InvalidBase64);
    }

    let bytes = payload.as_bytes();
    let padding = if bytes.ends_with(b"==") {
        2
    } else if bytes.ends_with(b"=") {
        1
    } else {
        0
    };
    let alphabet_len = bytes.len().saturating_sub(padding);
    if bytes[..alphabet_len].iter().any(|byte| {
        !byte.is_ascii_alphanumeric() && *byte != b'+' && *byte != b'/'
    }) || bytes[alphabet_len..].iter().any(|byte| *byte != b'=')
    {
        return Err(AmlReject::InvalidBase64);
    }
    if padding > 2 {
        return Err(AmlReject::InvalidBase64);
    }
    Ok((bytes.len() / 4) * 3 - padding)
}

fn validate_content(
    track: Track,
    content: &MessageContent,
) -> Result<usize, AmlReject> {
    match content {
        MessageContent::Text(text) => {
            if text.trim().is_empty() {
                return Err(AmlReject::EmptyText);
            }
            Ok(0)
        }
        MessageContent::Parts(parts) => {
            if track != Track::Multimodal {
                return Err(AmlReject::TrackContentMismatch);
            }
            if parts.is_empty() {
                return Err(AmlReject::EmptyContentParts);
            }
            let mut total_image_bytes = 0usize;
            for part in parts {
                match part {
                    ContentPart::Text(text) if text.trim().is_empty() => {
                        return Err(AmlReject::EmptyText);
                    }
                    ContentPart::Text(_) => {}
                    ContentPart::ImageDataUri(uri) => {
                        let decoded = image_data_uri_decoded_len(uri)?;
                        if decoded > MAX_IMAGE_BYTES {
                            return Err(AmlReject::ImageTooLarge);
                        }
                        total_image_bytes = total_image_bytes.saturating_add(decoded);
                    }
                }
            }
            Ok(total_image_bytes)
        }
    }
}

fn validate_message(track: Track, message: &Message) -> Result<usize, AmlReject> {
    if message.timestamp_ms.is_some_and(|value| value < 0) {
        return Err(AmlReject::InvalidTimestamp);
    }
    if track == Track::Textual || track == Track::Coding {
        if !matches!(&message.content, MessageContent::Text(_)) {
            return Err(AmlReject::TrackContentMismatch);
        }
    }
    validate_content(track, &message.content)
}

pub fn validate_add_request(track: Track, request: &AddRequest) -> Result<(), AmlReject> {
    if request.request_id.trim().is_empty() {
        return Err(AmlReject::EmptyRequestId);
    }
    if request.user_id.trim().is_empty() {
        return Err(AmlReject::EmptyUserId);
    }
    if request.session_id.trim().is_empty() {
        return Err(AmlReject::EmptySessionId);
    }
    if request.messages.is_empty() {
        return Err(AmlReject::EmptyMessages);
    }
    if request.messages.len() > TEXTUAL_MAX_MESSAGES && track == Track::Textual {
        return Err(AmlReject::TooManyMessages);
    }
    let mut image_bytes = 0usize;
    for message in &request.messages {
        image_bytes = image_bytes.saturating_add(validate_message(track, message)?);
    }
    let _ = image_bytes;
    Ok(())
}

/// The frozen AML adapter's exact word-counting implementation is not fully
/// specified in the public API text. Inject it explicitly; never substitute an
/// unverified local tokenizer when producing a leaderboard submission.
pub trait FrozenAdapterWordCounter {
    fn adapter_version(&self) -> &str;
    fn count_words(&self, messages: &[Message]) -> usize;
}

/// Validate one Textual Add chunk using the frozen adapter's counter.
pub fn validate_textual_chunk(
    request: &AddRequest,
    counter: &impl FrozenAdapterWordCounter,
) -> Result<(), AmlReject> {
    validate_add_request(Track::Textual, request)?;
    if counter.adapter_version().trim().is_empty() {
        return Err(AmlReject::WordCounterVersionMissing);
    }
    if counter.count_words(&request.messages) > TEXTUAL_MAX_WORDS {
        return Err(AmlReject::TooManyWords);
    }
    Ok(())
}

pub fn validate_search_request_for_track(
    track: Track,
    request: &SearchRequest,
) -> Result<(), AmlReject> {
    validate_search_request(request)?;
    if track != Track::Multimodal
        && !matches!(&request.query, MessageContent::Text(_))
    {
        return Err(AmlReject::TrackContentMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextualChunkPlan {
    pub word_counter_version: String,
    pub chunks: Vec<AddRequest>,
}

/// Deterministically chunk Textual messages in source order at either 20
/// messages or 2,000 words, using the injected frozen-counter implementation.
pub fn chunk_textual_messages(
    request: &AddRequest,
    counter: &impl FrozenAdapterWordCounter,
) -> Result<TextualChunkPlan, AmlReject> {
    if request.request_id.trim().is_empty() {
        return Err(AmlReject::EmptyRequestId);
    }
    if request.user_id.trim().is_empty() {
        return Err(AmlReject::EmptyUserId);
    }
    if request.session_id.trim().is_empty() {
        return Err(AmlReject::EmptySessionId);
    }
    if request.messages.is_empty() {
        return Err(AmlReject::EmptyMessages);
    }
    if counter.adapter_version().trim().is_empty() {
        return Err(AmlReject::WordCounterVersionMissing);
    }
    for message in &request.messages {
        validate_message(Track::Textual, message)?;
        if counter.count_words(std::slice::from_ref(message)) > TEXTUAL_MAX_WORDS {
            return Err(AmlReject::SingleMessageWordLimitExceeded);
        }
    }

    let mut chunks = Vec::new();
    let mut current = Vec::<Message>::new();
    for message in &request.messages {
        let mut candidate = current.clone();
        candidate.push(message.clone());
        let over_count = candidate.len() > TEXTUAL_MAX_MESSAGES;
        let over_words = counter.count_words(&candidate) > TEXTUAL_MAX_WORDS;
        if !current.is_empty() && (over_count || over_words) {
            chunks.push(current);
            current = vec![message.clone()];
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    let chunks = chunks
        .into_iter()
        .enumerate()
        .map(|(index, messages)| AddRequest {
            request_id: format!("{}:chunk-{index}", request.request_id),
            messages,
            user_id: request.user_id.clone(),
            session_id: request.session_id.clone(),
        })
        .collect();

    Ok(TextualChunkPlan {
        word_counter_version: counter.adapter_version().to_owned(),
        chunks,
    })
}

pub fn validate_add_response(
    request: &AddRequest,
    response: &AddResponse,
    persisted_and_searchable: bool,
) -> Result<(), AmlReject> {
    if !persisted_and_searchable {
        return Err(AmlReject::AddNotPersistedAndSearchable);
    }
    if !response.success {
        return Err(AmlReject::AddResponseNotSuccessful);
    }
    if response.request_id != request.request_id
        || response.user_id != request.user_id
        || response.session_id != request.session_id
    {
        return Err(AmlReject::AddResponseEchoMismatch);
    }
    Ok(())
}

/// Request-body identity for idempotent Add retries. Length-prefixing prevents
/// delimiter ambiguity; a request ID cannot be rebound to different content.
pub fn add_payload_digest(request: &AddRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"aml:add-request:v1\0");
    hash_field(&mut hasher, &request.request_id);
    hash_field(&mut hasher, &request.user_id);
    hash_field(&mut hasher, &request.session_id);
    hasher.update((request.messages.len() as u64).to_be_bytes());
    for message in &request.messages {
        hash_field(&mut hasher, message.role.as_str());
        hash_field(
            &mut hasher,
            &message.timestamp_ms.unwrap_or(i64::MIN).to_string(),
        );
        match &message.content {
            MessageContent::Text(text) => {
                hash_field(&mut hasher, "text");
                hash_field(&mut hasher, text);
            }
            MessageContent::Parts(parts) => {
                hash_field(&mut hasher, "parts");
                hasher.update((parts.len() as u64).to_be_bytes());
                for part in parts {
                    match part {
                        ContentPart::Text(text) => {
                            hash_field(&mut hasher, "text");
                            hash_field(&mut hasher, text);
                        }
                        ContentPart::ImageDataUri(uri) => {
                            hash_field(&mut hasher, "image_url");
                            hash_field(&mut hasher, uri);
                        }
                    }
                }
            }
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AddRetryState {
    New,
    InFlight,
    Committed(AddResponse),
}

/// In-memory contract helper only. A hosted service must persist this state
/// atomically with the memory write in its durable store/transaction.
#[derive(Clone, Debug, Default)]
pub struct AddIdempotencyLedger {
    entries: BTreeMap<String, (String, Option<AddResponse>)>,
}

impl AddIdempotencyLedger {
    pub fn begin(
        &mut self,
        track: Track,
        request: &AddRequest,
    ) -> Result<AddRetryState, AmlReject> {
        validate_add_request(track, request)?;
        let digest = add_payload_digest(request);
        match self.entries.get(&request.request_id) {
            Some((existing, Some(response))) if existing == &digest => {
                Ok(AddRetryState::Committed(response.clone()))
            }
            Some((existing, None)) if existing == &digest => Ok(AddRetryState::InFlight),
            Some(_) => Err(AmlReject::RequestIdPayloadConflict),
            None => {
                self.entries
                    .insert(request.request_id.clone(), (digest, None));
                Ok(AddRetryState::New)
            }
        }
    }

    pub fn complete(
        &mut self,
        track: Track,
        request: &AddRequest,
        response: AddResponse,
        persisted_and_searchable: bool,
    ) -> Result<(), AmlReject> {
        validate_add_request(track, request)?;
        validate_add_response(request, &response, persisted_and_searchable)?;
        let digest = add_payload_digest(request);
        let Some((stored_digest, stored_response)) = self.entries.get_mut(&request.request_id)
        else {
            return Err(AmlReject::UnknownRequestId);
        };
        if *stored_digest != digest {
            return Err(AmlReject::RequestIdPayloadConflict);
        }
        if stored_response.is_some() {
            return Err(AmlReject::AddAlreadyCompleted);
        }
        *stored_response = Some(response);
        Ok(())
    }

    /// Remove only a matching in-flight reservation after a definite failure.
    /// An unknown/committed/different payload is never removed.
    pub fn abandon(&mut self, request: &AddRequest) -> Result<(), AmlReject> {
        let digest = add_payload_digest(request);
        match self.entries.get(&request.request_id) {
            Some((stored, None)) if stored == &digest => {
                self.entries.remove(&request.request_id);
                Ok(())
            }
            Some((stored, _)) if stored != &digest => {
                Err(AmlReject::RequestIdPayloadConflict)
            }
            Some((_, Some(_))) => Err(AmlReject::AddAlreadyCompleted),
            None => Err(AmlReject::UnknownRequestId),
        }
    }
}

pub fn validate_search_request(request: &SearchRequest) -> Result<(), AmlReject> {
    if request.user_id.trim().is_empty() {
        return Err(AmlReject::EmptySearchUserId);
    }
    if request.top_k == 0 || request.top_k > FORMAL_TOP_K {
        return Err(AmlReject::InvalidTopK);
    }
    match &request.query {
        MessageContent::Text(text) if text.trim().is_empty() => {
            return Err(AmlReject::EmptyText);
        }
        MessageContent::Text(_) => {}
        MessageContent::Parts(parts) => {
            validate_content(Track::Multimodal, &MessageContent::Parts(parts.clone()))?;
        }
    }
    if let Some(options) = &request.options {
        if options.is_empty() {
            return Err(AmlReject::EmptyOptions);
        }
        if options.iter().any(|option| option.trim().is_empty()) {
            return Err(AmlReject::EmptyOption);
        }
    }
    Ok(())
}

pub fn validate_search_response(
    request: &SearchRequest,
    response: &SearchResponse,
) -> Result<(), AmlReject> {
    validate_search_response_for_track(Track::Textual, request, response)
}

pub fn validate_search_response_for_track(
    track: Track,
    request: &SearchRequest,
    response: &SearchResponse,
) -> Result<(), AmlReject> {
    validate_search_request_for_track(track, request)?;
    if response.data.len() > request.top_k {
        return Err(AmlReject::SearchResponseTooLarge);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut image_bytes = 0usize;
    for item in &response.data {
        if item.id.trim().is_empty() {
            return Err(AmlReject::EmptyMemoryId);
        }
        if !seen.insert(item.id.as_str()) {
            return Err(AmlReject::DuplicateMemoryId);
        }
        if item.score.is_some_and(|score| !score.is_finite()) {
            return Err(AmlReject::InvalidScore);
        }
        if let Some(created_at) = &item.created_at {
            if created_at.trim().is_empty() {
                return Err(AmlReject::EmptyCreatedAt);
            }
        }
        if let MessageContent::Text(content) = &item.content {
            if content.trim().is_empty() {
                return Err(AmlReject::EmptyMemoryContent);
            }
        }
        let item_track = match (&item.content, track) {
            (MessageContent::Text(_), _) => track,
            (MessageContent::Parts(_), Track::Multimodal) => Track::Multimodal,
            (MessageContent::Parts(_), _) => {
                return Err(AmlReject::TrackContentMismatch);
            }
        };
        image_bytes = image_bytes.saturating_add(validate_content(item_track, &item.content)?);
    }
    if track == Track::Multimodal && image_bytes > MAX_SEARCH_RESPONSE_IMAGE_BYTES {
        return Err(AmlReject::SearchImageBudgetExceeded);
    }
    Ok(())
}

impl AddRequest {
    pub fn to_json_value(&self) -> Value {
        let messages = self
            .messages
            .iter()
            .map(|message| {
                let mut value = json!({
                    "role": message.role.as_str(),
                    "content": content_to_json(&message.content)
                });
                if let Some(timestamp) = message.timestamp_ms {
                    value["timestamp"] = json!(timestamp);
                }
                value
            })
            .collect::<Vec<_>>();
        json!({
            "request_id": self.request_id,
            "messages": messages,
            "user_id": self.user_id,
            "session_id": self.session_id
        })
    }
}

fn content_to_json(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => json!(text),
        MessageContent::Parts(parts) => Value::Array(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => json!({"type": "text", "text": text}),
                    ContentPart::ImageDataUri(uri) => {
                        json!({"type": "image_url", "image_url": {"url": uri}})
                    }
                })
                .collect(),
        ),
    }
}

impl SearchRequest {
    pub fn to_json_value(&self) -> Value {
        let mut request = json!({
            "query": content_to_json(&self.query),
            "user_id": self.user_id,
            "top_k": self.top_k
        });
        if let Some(options) = &self.options {
            request["options"] = json!(options);
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WhitespaceCounter;

    impl FrozenAdapterWordCounter for WhitespaceCounter {
        fn adapter_version(&self) -> &str {
            "TEST-ONLY-whitespace-counter-v1"
        }

        fn count_words(&self, messages: &[Message]) -> usize {
            messages
                .iter()
                .map(|message| match &message.content {
                    MessageContent::Text(text) => text.split_whitespace().count(),
                    MessageContent::Parts(_) => 0,
                })
                .sum()
        }
    }

    fn message(role: Role, text: &str) -> Message {
        Message {
            role,
            timestamp_ms: Some(1_700_000_000_000),
            content: MessageContent::Text(text.into()),
        }
    }

    fn add_request(request_id: &str) -> AddRequest {
        AddRequest {
            request_id: request_id.into(),
            messages: vec![message(Role::User, "remember this")],
            user_id: "eval:user:one".into(),
            session_id: "eval:session:one".into(),
        }
    }

    fn ack(request: &AddRequest) -> AddResponse {
        AddResponse {
            success: true,
            request_id: request.request_id.clone(),
            user_id: request.user_id.clone(),
            session_id: request.session_id.clone(),
        }
    }

    #[test]
    fn add_request_validates_track_role_content_and_ids() {
        let request = add_request("write-1");
        assert_eq!(validate_add_request(Track::Textual, &request), Ok(()));

        let mut wrong_track = request.clone();
        wrong_track.messages[0].content =
            MessageContent::Parts(vec![ContentPart::Text("x".into())]);
        assert_eq!(
            validate_add_request(Track::Textual, &wrong_track),
            Err(AmlReject::TrackContentMismatch)
        );
    }

    #[test]
    fn add_ack_requires_persisted_searchable_write_and_exact_echo() {
        let request = add_request("write-2");
        assert_eq!(
            validate_add_response(&request, &ack(&request), false),
            Err(AmlReject::AddNotPersistedAndSearchable)
        );
        let mut mismatch = ack(&request);
        mismatch.user_id = "another-user".into();
        assert_eq!(
            validate_add_response(&request, &mismatch, true),
            Err(AmlReject::AddResponseEchoMismatch)
        );
        assert_eq!(validate_add_response(&request, &ack(&request), true), Ok(()));
    }

    #[test]
    fn add_idempotency_replays_exact_request_and_rejects_payload_rebinding() {
        let request = add_request("write-3");
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(ledger.begin(Track::Textual, &request), Ok(AddRetryState::New));
        assert_eq!(ledger.begin(Track::Textual, &request), Ok(AddRetryState::InFlight));
        ledger.complete(Track::Textual, &request, ack(&request), true).unwrap();
        assert_eq!(
            ledger.begin(Track::Textual, &request),
            Ok(AddRetryState::Committed(ack(&request)))
        );

        let mut changed = request.clone();
        changed.messages[0].content = MessageContent::Text("different".into());
        assert_eq!(
            ledger.begin(Track::Textual, &changed),
            Err(AmlReject::RequestIdPayloadConflict)
        );
    }

    #[test]
    fn textual_chunking_preserves_order_and_respects_both_limits() {
        let messages = (0..21)
            .map(|index| message(Role::User, &format!("message {index}")))
            .collect::<Vec<_>>();
        let request = AddRequest {
            request_id: "logical-write".into(),
            messages,
            user_id: "user".into(),
            session_id: "session".into(),
        };
        let plan = chunk_textual_messages(&request, &WhitespaceCounter).unwrap();
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].messages.len(), 20);
        assert_eq!(plan.chunks[1].messages.len(), 1);
        assert_eq!(plan.chunks[0].request_id, "logical-write:chunk-0");
        assert_eq!(plan.chunks[1].messages[0].content, MessageContent::Text("message 20".into()));
    }

    #[test]
    fn textual_chunking_splits_at_word_boundary_with_frozen_counter() {
        let request = AddRequest {
            request_id: "words".into(),
            messages: vec![
                message(Role::User, &vec!["a"; 1200].join(" ")),
                message(Role::Assistant, &vec!["b"; 900].join(" ")),
            ],
            user_id: "user".into(),
            session_id: "session".into(),
        };
        let plan = chunk_textual_messages(&request, &WhitespaceCounter).unwrap();
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].messages.len(), 1);
    }

    #[test]
    fn idempotency_abandon_only_removes_matching_inflight_payload() {
        let request = add_request("write-4");
        let mut ledger = AddIdempotencyLedger::default();
        ledger.begin(Track::Textual, &request).unwrap();
        assert_eq!(ledger.abandon(&request), Ok(()));
        assert_eq!(
            ledger.begin(Track::Textual, &request),
            Ok(AddRetryState::New)
        );
    }

    #[test]
    fn idempotency_reservation_validates_request_before_claiming_request_id() {
        let mut invalid = add_request("");
        invalid.user_id = " ".into();
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(
            ledger.begin(Track::Textual, &invalid),
            Err(AmlReject::EmptyRequestId)
        );
        invalid.request_id = "valid-id".into();
        assert_eq!(
            ledger.begin(Track::Textual, &invalid),
            Err(AmlReject::EmptyUserId)
        );
    }

    #[test]
    fn search_request_and_response_preserve_user_scope_and_rank_order() {
        let request = SearchRequest {
            query: MessageContent::Text("what did I ask?".into()),
            options: Some(vec!["A".into(), "B".into()]),
            user_id: "eval:user:one".into(),
            top_k: FORMAL_TOP_K,
        };
        let response = SearchResponse {
            data: vec![
                SearchItem {
                    id: "mem-1".into(),
                    content: MessageContent::Text("first".into()),
                    score: Some(0.9),
                    created_at: None,
                },
                SearchItem {
                    id: "mem-2".into(),
                    content: MessageContent::Text("second".into()),
                    score: Some(0.8),
                    created_at: None,
                },
            ],
        };
        assert_eq!(validate_search_response(&request, &response), Ok(()));
        let json = request.to_json_value();
        assert_eq!(json["user_id"], "eval:user:one");
        assert_eq!(json["top_k"], 100);
        assert_eq!(json["options"][0], "A");
    }

    #[test]
    fn search_rejects_cross_scope_isolation_misuse_and_over_topk() {
        let request = SearchRequest {
            query: MessageContent::Text("query".into()),
            options: None,
            user_id: "user".into(),
            top_k: 1,
        };
        let mut response = SearchResponse {
            data: vec![
                SearchItem {
                    id: "one".into(),
                    content: MessageContent::Text("one".into()),
                    score: None,
                    created_at: None,
                },
                SearchItem {
                    id: "two".into(),
                    content: MessageContent::Text("two".into()),
                    score: None,
                    created_at: None,
                },
            ],
        };
        assert_eq!(
            validate_search_response(&request, &response),
            Err(AmlReject::SearchResponseTooLarge)
        );
        response.data.truncate(1);
        assert_eq!(validate_search_response(&request, &response), Ok(()));
    }

    #[test]
    fn multimodal_data_uris_are_validated_and_remote_urls_rejected() {
        let request = AddRequest {
            request_id: "multi".into(),
            user_id: "user".into(),
            session_id: "session".into(),
            messages: vec![Message {
                role: Role::User,
                timestamp_ms: None,
                content: MessageContent::Parts(vec![
                    ContentPart::Text("describe this".into()),
                    ContentPart::ImageDataUri("data:image/png;base64,aGVsbG8=".into()),
                ]),
            }],
        };
        assert_eq!(validate_add_request(Track::Multimodal, &request), Ok(()));
        let mut invalid = request;
        invalid.messages[0].content = MessageContent::Parts(vec![
            ContentPart::ImageDataUri("https://example.com/image.png".into()),
        ]);
        assert_eq!(
            validate_add_request(Track::Multimodal, &invalid),
            Err(AmlReject::InvalidImageDataUri)
        );
    }

    #[test]
    fn textual_add_chunk_enforces_the_injected_frozen_word_counter() {
        let request = AddRequest {
            request_id: "word-limit".into(),
            messages: vec![message(Role::User, &vec!["x"; 2_001].join(" "))],
            user_id: "user".into(),
            session_id: "session".into(),
        };
        assert_eq!(
            validate_textual_chunk(&request, &WhitespaceCounter),
            Err(AmlReject::TooManyWords)
        );
    }

    #[test]
    fn track_specific_search_rejects_multimodal_payload_for_textual_track() {
        let request = SearchRequest {
            query: MessageContent::Parts(vec![
                ContentPart::Text("describe".into()),
                ContentPart::ImageDataUri("data:image/png;base64,aGVsbG8=".into()),
            ]),
            options: None,
            user_id: "user".into(),
            top_k: FORMAL_TOP_K,
        };
        assert_eq!(
            validate_search_request_for_track(Track::Textual, &request),
            Err(AmlReject::TrackContentMismatch)
        );
        assert_eq!(
            validate_search_request_for_track(Track::Multimodal, &request),
            Ok(())
        );
    }

    #[test]
    fn formal_search_rejects_invalid_topk_and_empty_data_stays_valid() {
        let request = SearchRequest {
            query: MessageContent::Text("query".into()),
            options: None,
            user_id: "user".into(),
            top_k: 101,
        };
        assert_eq!(validate_search_request(&request), Err(AmlReject::InvalidTopK));
        let valid = SearchRequest { top_k: 100, ..request };
        assert_eq!(
            validate_search_response(&valid, &SearchResponse { data: Vec::new() }),
            Ok(())
        );
    }
}
