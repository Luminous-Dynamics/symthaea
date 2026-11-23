pub mod submission;
pub mod raw_session;
pub mod longmemeval_v2;
pub mod candidate_evidence;
pub mod longmemeval_v2_candidate_adapter;
pub mod retrieval_policy_memory;
pub mod retrieval_policy_store;

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
pub const MAX_ADD_REQUEST_IMAGE_BYTES: usize = 30 * 1024 * 1024;
pub const MAX_SEARCH_RESPONSE_IMAGE_BYTES: usize = 30 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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
pub struct AddVisibilityReceiptV1 {
    pub track: Track,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
    pub request_payload_digest: String,
    pub write_receipt_digest: String,
    pub search_visibility_receipt_digest: String,
    pub persisted_at_ms: i64,
    pub searchable_at_ms: i64,
    pub visible_memory_ids: Vec<String>,
}

pub trait AddVisibilityVerifierV1 {
    fn verify(&self, receipt: &AddVisibilityReceiptV1) -> Result<(), AmlReject>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedAddVisibilityV1 {
    receipt: AddVisibilityReceiptV1,
    response: AddResponse,
    word_counter_version: Option<String>,
}

impl VerifiedAddVisibilityV1 {
    pub fn receipt(&self) -> &AddVisibilityReceiptV1 {
        &self.receipt
    }

    pub fn visible_memory_ids(&self) -> &[String] {
        &self.receipt.visible_memory_ids
    }

    pub fn request_payload_digest(&self) -> &str {
        &self.receipt.request_payload_digest
    }

    pub fn word_counter_version(&self) -> Option<&str> {
        self.word_counter_version.as_deref()
    }
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
    AddVisibilityBindingMismatch,
    AddVisibilityInvalidReceiptDigest,
    AddVisibilityInvalidTimeOrder,
    AddVisibilityNoSearchableItems,
    AddVisibilityDuplicateMemoryId,
    AddVisibilityVerificationFailed,
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
    WordCounterVersionMismatch,
    SingleMessageWordLimitExceeded,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn image_data_uri_decoded_len(value: &str) -> Result<usize, AmlReject> {
    let (header, payload) = value
        .split_once(',')
        .ok_or(AmlReject::InvalidImageDataUri)?;
    if !matches!(
        header,
        "data:image/jpeg;base64" | "data:image/png;base64" | "data:image/webp;base64"
    ) {
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
    if padding > 2
        || (padding == 1 && alphabet_len % 4 != 3)
        || (padding == 2 && alphabet_len % 4 != 2)
        || (padding == 0 && alphabet_len % 4 != 0)
    {
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
    if track == Track::Multimodal && image_bytes > MAX_ADD_REQUEST_IMAGE_BYTES {
        return Err(AmlReject::SearchImageBudgetExceeded);
    }
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
    track: Track,
    request: &AddRequest,
    response: &AddResponse,
    receipt: &AddVisibilityReceiptV1,
    verifier: &impl AddVisibilityVerifierV1,
) -> Result<VerifiedAddVisibilityV1, AmlReject> {
    if track == Track::Textual {
        return Err(AmlReject::WordCounterVersionMissing);
    }
    validate_add_response_inner(track, request, response, receipt, verifier, None)
}

/// Textual visibility verification is permitted only after the exact frozen
/// adapter word counter and per-chunk limits have been checked.
pub fn validate_textual_add_response(
    request: &AddRequest,
    response: &AddResponse,
    receipt: &AddVisibilityReceiptV1,
    verifier: &impl AddVisibilityVerifierV1,
    counter: &impl FrozenAdapterWordCounter,
) -> Result<VerifiedAddVisibilityV1, AmlReject> {
    validate_textual_chunk(request, counter)?;
    validate_add_response_inner(
        Track::Textual,
        request,
        response,
        receipt,
        verifier,
        Some(counter.adapter_version()),
    )
}

fn validate_add_response_inner(
    track: Track,
    request: &AddRequest,
    response: &AddResponse,
    receipt: &AddVisibilityReceiptV1,
    verifier: &impl AddVisibilityVerifierV1,
    word_counter_version: Option<&str>,
) -> Result<VerifiedAddVisibilityV1, AmlReject> {
    validate_add_request(track, request)?;
    if !response.success {
        return Err(AmlReject::AddResponseNotSuccessful);
    }
    if response.request_id != request.request_id
        || response.user_id != request.user_id
        || response.session_id != request.session_id
    {
        return Err(AmlReject::AddResponseEchoMismatch);
    }

    let payload_digest = add_payload_digest(track, request);
    if receipt.track != track
        || receipt.request_id != request.request_id
        || receipt.user_id != request.user_id
        || receipt.session_id != request.session_id
        || receipt.request_payload_digest != payload_digest
    {
        return Err(AmlReject::AddVisibilityBindingMismatch);
    }
    if !is_lower_sha256(&receipt.write_receipt_digest)
        || !is_lower_sha256(&receipt.search_visibility_receipt_digest)
    {
        return Err(AmlReject::AddVisibilityInvalidReceiptDigest);
    }
    if receipt.persisted_at_ms < 0
        || receipt.searchable_at_ms < receipt.persisted_at_ms
    {
        return Err(AmlReject::AddVisibilityInvalidTimeOrder);
    }
    if receipt.visible_memory_ids.is_empty() {
        return Err(AmlReject::AddVisibilityNoSearchableItems);
    }
    let mut seen = std::collections::BTreeSet::new();
    for memory_id in &receipt.visible_memory_ids {
        if memory_id.trim().is_empty() {
            return Err(AmlReject::AddVisibilityNoSearchableItems);
        }
        if !seen.insert(memory_id.as_str()) {
            return Err(AmlReject::AddVisibilityDuplicateMemoryId);
        }
    }
    verifier
        .verify(receipt)
        .map_err(|_| AmlReject::AddVisibilityVerificationFailed)?;

    Ok(VerifiedAddVisibilityV1 {
        receipt: receipt.clone(),
        response: response.clone(),
        word_counter_version: word_counter_version.map(str::to_owned),
    })
}

/// Request-body identity for idempotent Add retries. Length-prefixing prevents
/// delimiter ambiguity; a request ID cannot be rebound to different content.
fn idempotency_payload_digest(
    track: Track,
    request: &AddRequest,
    word_counter_version: Option<&str>,
) -> String {
    let payload = add_payload_digest(track, request);
    let version = word_counter_version.unwrap_or("");
    let mut hasher = Sha256::new();
    hasher.update(b"aml:add-idempotency:v1\\0");
    hash_field(&mut hasher, &payload);
    hash_field(&mut hasher, version);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn add_payload_digest(track: Track, request: &AddRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"aml:add-request:v2\0");
    hasher.update(match track {
        Track::Textual => b"textual\0".as_slice(),
        Track::Coding => b"coding\0".as_slice(),
        Track::Multimodal => b"multimodal\0".as_slice(),
    });
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
    Committed(VerifiedAddVisibilityV1),
}

/// In-memory contract helper only. A hosted service must persist this state
/// atomically with the memory write in its durable store/transaction.
#[derive(Clone, Debug, Default)]
pub struct AddIdempotencyLedger {
    entries: BTreeMap<(Track, String, String), (String, Option<VerifiedAddVisibilityV1>)>,
}

fn idempotency_key(track: Track, request: &AddRequest) -> (Track, String, String) {
    (
        track,
        request.user_id.clone(),
        request.request_id.clone(),
    )
}

impl AddIdempotencyLedger {
    /// Non-Textual reservation. Textual requests must use `begin_textual` so
    /// the frozen adapter word counter is part of the reservation identity.
    pub fn begin(
        &mut self,
        track: Track,
        request: &AddRequest,
    ) -> Result<AddRetryState, AmlReject> {
        if track == Track::Textual {
            return Err(AmlReject::WordCounterVersionMissing);
        }
        self.begin_validated(track, request, None)
    }

    pub fn begin_textual(
        &mut self,
        request: &AddRequest,
        counter: &impl FrozenAdapterWordCounter,
    ) -> Result<AddRetryState, AmlReject> {
        validate_textual_chunk(request, counter)?;
        self.begin_validated(Track::Textual, request, Some(counter.adapter_version()))
    }

    fn begin_validated(
        &mut self,
        track: Track,
        request: &AddRequest,
        word_counter_version: Option<&str>,
    ) -> Result<AddRetryState, AmlReject> {
        validate_add_request(track, request)?;
        let digest = idempotency_payload_digest(track, request, word_counter_version);
        let key = idempotency_key(track, request);
        match self.entries.get(&key) {
            Some((existing, Some(receipt))) if existing == &digest => {
                Ok(AddRetryState::Committed(receipt.clone()))
            }
            Some((existing, None)) if existing == &digest => Ok(AddRetryState::InFlight),
            Some(_) => Err(AmlReject::RequestIdPayloadConflict),
            None => {
                self.entries.insert(key, (digest, None));
                Ok(AddRetryState::New)
            }
        }
    }

    pub fn complete(
        &mut self,
        track: Track,
        request: &AddRequest,
        verified: VerifiedAddVisibilityV1,
    ) -> Result<(), AmlReject> {
        if track == Track::Textual {
            return Err(AmlReject::WordCounterVersionMissing);
        }
        self.complete_validated(track, request, verified, None)
    }

    pub fn complete_textual(
        &mut self,
        request: &AddRequest,
        verified: VerifiedAddVisibilityV1,
        counter: &impl FrozenAdapterWordCounter,
    ) -> Result<(), AmlReject> {
        validate_textual_chunk(request, counter)?;
        self.complete_validated(
            Track::Textual,
            request,
            verified,
            Some(counter.adapter_version()),
        )
    }

    fn complete_validated(
        &mut self,
        track: Track,
        request: &AddRequest,
        verified: VerifiedAddVisibilityV1,
        word_counter_version: Option<&str>,
    ) -> Result<(), AmlReject> {
        validate_add_request(track, request)?;
        if verified.word_counter_version.as_deref() != word_counter_version {
            return Err(AmlReject::WordCounterVersionMismatch);
        }
        let receipt = verified.receipt();
        let digest = idempotency_payload_digest(track, request, word_counter_version);
        if receipt.track != track
            || receipt.request_id != request.request_id
            || receipt.user_id != request.user_id
            || receipt.session_id != request.session_id
            || receipt.request_payload_digest != add_payload_digest(track, request)
        {
            return Err(AmlReject::AddVisibilityBindingMismatch);
        }
        let key = idempotency_key(track, request);
        let Some((stored_digest, stored_receipt)) = self.entries.get_mut(&key)
        else {
            return Err(AmlReject::UnknownRequestId);
        };
        if *stored_digest != digest {
            return Err(AmlReject::RequestIdPayloadConflict);
        }
        if stored_receipt.is_some() {
            return Err(AmlReject::AddAlreadyCompleted);
        }
        *stored_receipt = Some(verified);
        Ok(())
    }

    pub fn abandon(
        &mut self,
        track: Track,
        request: &AddRequest,
    ) -> Result<(), AmlReject> {
        if track == Track::Textual {
            return Err(AmlReject::WordCounterVersionMissing);
        }
        self.abandon_validated(track, request, None)
    }

    pub fn abandon_textual(
        &mut self,
        request: &AddRequest,
        counter: &impl FrozenAdapterWordCounter,
    ) -> Result<(), AmlReject> {
        validate_textual_chunk(request, counter)?;
        self.abandon_validated(Track::Textual, request, Some(counter.adapter_version()))
    }

    fn abandon_validated(
        &mut self,
        track: Track,
        request: &AddRequest,
        word_counter_version: Option<&str>,
    ) -> Result<(), AmlReject> {
        let digest = idempotency_payload_digest(track, request, word_counter_version);
        let key = idempotency_key(track, request);
        match self.entries.get(&key) {
            Some((stored, None)) if stored == &digest => {
                self.entries.remove(&key);
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

    struct TestAddVisibilityVerifier;

    impl AddVisibilityVerifierV1 for TestAddVisibilityVerifier {
        fn verify(&self, receipt: &AddVisibilityReceiptV1) -> Result<(), AmlReject> {
            let expected_write = sha256_hex(
                format!(
                    "fixture-write:{}:{}",
                    receipt.request_payload_digest,
                    receipt.persisted_at_ms
                )
                .as_bytes(),
            );
            let expected_search = sha256_hex(
                format!(
                    "fixture-search:{}:{}",
                    receipt.request_payload_digest,
                    receipt.searchable_at_ms
                )
                .as_bytes(),
            );
            if receipt.write_receipt_digest != expected_write
                || receipt.search_visibility_receipt_digest != expected_search
            {
                return Err(AmlReject::AddVisibilityVerificationFailed);
            }
            Ok(())
        }
    }

    fn visibility_receipt(
        track: Track,
        request: &AddRequest,
    ) -> AddVisibilityReceiptV1 {
        let digest = add_payload_digest(track, request);
        let persisted_at_ms = 10;
        let searchable_at_ms = 11;
        AddVisibilityReceiptV1 {
            track,
            request_id: request.request_id.clone(),
            user_id: request.user_id.clone(),
            session_id: request.session_id.clone(),
            request_payload_digest: digest.clone(),
            write_receipt_digest: sha256_hex(
                format!("fixture-write:{digest}:{persisted_at_ms}").as_bytes(),
            ),
            search_visibility_receipt_digest: sha256_hex(
                format!("fixture-search:{digest}:{searchable_at_ms}").as_bytes(),
            ),
            persisted_at_ms,
            searchable_at_ms,
            visible_memory_ids: vec!["memory-1".into()],
        }
    }

    fn verified_add(
        track: Track,
        request: &AddRequest,
    ) -> VerifiedAddVisibilityV1 {
        let result = if track == Track::Textual {
            validate_textual_add_response(
                request,
                &ack(request),
                &visibility_receipt(track, request),
                &TestAddVisibilityVerifier,
                &WhitespaceCounter,
            )
        } else {
            validate_add_response(
                track,
                request,
                &ack(request),
                &visibility_receipt(track, request),
                &TestAddVisibilityVerifier,
            )
        };
        result.expect("fixture write/search receipts should verify")
    }

    fn begin_textual(
        ledger: &mut AddIdempotencyLedger,
        request: &AddRequest,
    ) -> Result<AddRetryState, AmlReject> {
        ledger.begin_textual(request, &WhitespaceCounter)
    }

    fn complete_textual(
        ledger: &mut AddIdempotencyLedger,
        request: &AddRequest,
        verified: VerifiedAddVisibilityV1,
    ) -> Result<(), AmlReject> {
        ledger.complete_textual(request, verified, &WhitespaceCounter)
    }

    fn abandon_textual(
        ledger: &mut AddIdempotencyLedger,
        request: &AddRequest,
    ) -> Result<(), AmlReject> {
        ledger.abandon_textual(request, &WhitespaceCounter)
    }

    struct AlternateWhitespaceCounter;

    impl FrozenAdapterWordCounter for AlternateWhitespaceCounter {
        fn adapter_version(&self) -> &str {
            "fixture-other-word-counter-v2"
        }

        fn count_words(&self, messages: &[Message]) -> usize {
            WhitespaceCounter.count_words(messages)
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
    fn add_ack_requires_exact_echo_and_verified_visibility() {
        let request = add_request("write-2");
        let receipt = visibility_receipt(Track::Textual, &request);
        let mut mismatch = ack(&request);
        mismatch.user_id = "another-user".into();
        assert_eq!(
            validate_textual_add_response(
        &request,
        &mismatch,
        &receipt,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
            .err(),
            Some(AmlReject::AddResponseEchoMismatch)
        );
        let verified = validate_textual_add_response(
        &request,
        &ack(&request),
        &receipt,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
        .unwrap();
        assert_eq!(verified.visible_memory_ids(), &["memory-1".to_owned()]);
    }

    #[test]
    fn add_visibility_receipt_rejects_binding_empty_visibility_bad_time_and_forgery() {
        let request = add_request("write-visibility");
        let mut wrong_user = visibility_receipt(Track::Textual, &request);
        wrong_user.user_id = "another-user".into();
        assert_eq!(
            validate_textual_add_response(
        &request,
        &ack(&request),
        &wrong_user,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
            .err(),
            Some(AmlReject::AddVisibilityBindingMismatch)
        );

        let mut empty = visibility_receipt(Track::Textual, &request);
        empty.visible_memory_ids.clear();
        assert_eq!(
            validate_textual_add_response(
        &request,
        &ack(&request),
        &empty,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
            .err(),
            Some(AmlReject::AddVisibilityNoSearchableItems)
        );

        let mut bad_time = visibility_receipt(Track::Textual, &request);
        bad_time.searchable_at_ms = 9;
        assert_eq!(
            validate_textual_add_response(
        &request,
        &ack(&request),
        &bad_time,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
            .err(),
            Some(AmlReject::AddVisibilityInvalidTimeOrder)
        );

        let mut forged = visibility_receipt(Track::Textual, &request);
        forged.search_visibility_receipt_digest = "0".repeat(64);
        assert_eq!(
            validate_textual_add_response(
        &request,
        &ack(&request),
        &forged,
        &TestAddVisibilityVerifier,
        &WhitespaceCounter,
    )
            .err(),
            Some(AmlReject::AddVisibilityVerificationFailed)
        );
    }

    #[test]
    fn add_idempotency_replays_exact_request_and_rejects_payload_rebinding() {
        let request = add_request("write-3");
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(begin_textual(&mut ledger, &request), Ok(AddRetryState::New));
        assert_eq!(begin_textual(&mut ledger, &request), Ok(AddRetryState::InFlight));
        ledger
            .complete(Track::Textual, &request, verified_add(Track::Textual, &request))
            .unwrap();
        assert_eq!(
            begin_textual(&mut ledger, &request),
            Ok(AddRetryState::Committed(verified_add(Track::Textual, &request)))
        );

        let mut changed = request.clone();
        changed.messages[0].content = MessageContent::Text("different".into());
        assert_eq!(
            begin_textual(&mut ledger, &changed),
            Err(AmlReject::RequestIdPayloadConflict)
        );
    }

    #[test]
    fn committed_idempotency_replay_preserves_visibility_receipt_evidence() {
        let request = add_request("receipt-replay");
        let original = verified_add(Track::Textual, &request);
        let expected_write = original.receipt().write_receipt_digest.clone();
        let expected_search = original
            .receipt()
            .search_visibility_receipt_digest
            .clone();
        let expected_ids = original.visible_memory_ids().to_vec();

        let mut ledger = AddIdempotencyLedger::default();
        begin_textual(&mut ledger, &request).unwrap();
        complete_textual(&mut ledger, &request, original).unwrap();

        let AddRetryState::Committed(replayed) =
            begin_textual(&mut ledger, &request).unwrap()
        else {
            panic!("committed retry must replay the verified receipt token");
        };
        assert_eq!(replayed.receipt().write_receipt_digest, expected_write);
        assert_eq!(
            replayed.receipt().search_visibility_receipt_digest,
            expected_search
        );
        assert_eq!(replayed.visible_memory_ids(), expected_ids);
    }

    #[test]
    fn idempotency_completion_rejects_visibility_token_from_another_scope() {
        let request = add_request("verified-token-scope");
        let mut other_user = request.clone();
        other_user.user_id = "eval:user:two".into();
        let mut ledger = AddIdempotencyLedger::default();
        begin_textual(&mut ledger, &request).unwrap();
        let foreign = verified_add(Track::Textual, &other_user);
        assert_eq!(
            complete_textual(&mut ledger, &request, foreign),
            Err(AmlReject::AddVisibilityBindingMismatch)
        );
        assert_eq!(
            begin_textual(&mut ledger, &request),
            Ok(AddRetryState::InFlight)
        );
    }

    #[test]
    fn textual_add_requires_frozen_counter_at_every_lifecycle_boundary() {
        let too_many_words = AddRequest {
            request_id: "too-many-words".into(),
            messages: vec![message(
                Role::User,
                &vec!["word"; TEXTUAL_MAX_WORDS + 1].join(" "),
            )],
            user_id: "eval:user:one".into(),
            session_id: "eval:session:one".into(),
        };
        let response = ack(&too_many_words);
        let receipt = visibility_receipt(Track::Textual, &too_many_words);

        assert_eq!(
            validate_add_response(
                Track::Textual,
                &too_many_words,
                &response,
                &receipt,
                &TestAddVisibilityVerifier,
            )
            .err(),
            Some(AmlReject::WordCounterVersionMissing)
        );
        assert_eq!(
            validate_textual_add_response(
                &too_many_words,
                &response,
                &receipt,
                &TestAddVisibilityVerifier,
                &WhitespaceCounter,
            )
            .err(),
            Some(AmlReject::TooManyWords)
        );
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(
            ledger.begin(Track::Textual, &too_many_words),
            Err(AmlReject::WordCounterVersionMissing)
        );
        assert_eq!(
            ledger.begin_textual(&too_many_words, &WhitespaceCounter),
            Err(AmlReject::TooManyWords)
        );

        let valid = add_request("textual-complete-bypass");
        let verified = verified_add(Track::Textual, &valid);
        let mut valid_ledger = AddIdempotencyLedger::default();
        valid_ledger
            .begin_textual(&valid, &WhitespaceCounter)
            .unwrap();
        assert_eq!(
            valid_ledger.complete(Track::Textual, &valid, verified.clone()),
            Err(AmlReject::WordCounterVersionMissing)
        );
        valid_ledger
            .complete_textual(&valid, verified, &WhitespaceCounter)
            .unwrap();
    }

    #[test]
    fn textual_idempotency_and_visibility_bind_the_frozen_counter_version() {
        let request = add_request("word-counter-version");
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(
            ledger.begin_textual(&request, &WhitespaceCounter),
            Ok(AddRetryState::New)
        );
        assert_eq!(
            ledger.begin_textual(&request, &AlternateWhitespaceCounter),
            Err(AmlReject::RequestIdPayloadConflict)
        );

        let foreign_version = validate_textual_add_response(
            &request,
            &ack(&request),
            &visibility_receipt(Track::Textual, &request),
            &TestAddVisibilityVerifier,
            &AlternateWhitespaceCounter,
        )
        .unwrap();
        assert_eq!(
            ledger.complete_textual(
                &request,
                foreign_version,
                &WhitespaceCounter,
            ),
            Err(AmlReject::WordCounterVersionMismatch)
        );
        assert_eq!(
            ledger.begin_textual(&request, &WhitespaceCounter),
            Ok(AddRetryState::InFlight)
        );

        let verified = verified_add(Track::Textual, &request);
        ledger
            .complete_textual(&request, verified, &WhitespaceCounter)
            .unwrap();
        assert!(matches!(
            ledger.begin_textual(&request, &WhitespaceCounter),
            Ok(AddRetryState::Committed(_))
        ));
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
    fn idempotency_keys_are_isolated_by_tenant_and_track() {
        let mut user_one = add_request("shared-request-id");
        let mut user_two = user_one.clone();
        user_two.user_id = "eval:user:two".into();
        let mut ledger = AddIdempotencyLedger::default();

        assert_eq!(
            begin_textual(&mut ledger, &user_one),
            Ok(AddRetryState::New)
        );
        assert_eq!(
            begin_textual(&mut ledger, &user_two),
            Ok(AddRetryState::New)
        );

        ledger
            .complete(
                Track::Textual,
                &user_one,
                verified_add(Track::Textual, &user_one),
            )
            .unwrap();
        ledger
            .complete(
                Track::Textual,
                &user_two,
                verified_add(Track::Textual, &user_two),
            )
            .unwrap();
        assert_eq!(
            begin_textual(&mut ledger, &user_one),
            Ok(AddRetryState::Committed(verified_add(Track::Textual, &user_one)))
        );
        assert_eq!(
            begin_textual(&mut ledger, &user_two),
            Ok(AddRetryState::Committed(verified_add(Track::Textual, &user_two)))
        );

        // Identical request ID, tenant and payload on a different track is
        // a different idempotency namespace and payload contract.
        let same_tenant_other_track = user_one.clone();
        assert_eq!(
            ledger.begin(Track::Coding, &same_tenant_other_track),
            Ok(AddRetryState::New)
        );
        ledger
            .complete(
                Track::Coding,
                &same_tenant_other_track,
                verified_add(Track::Coding, &same_tenant_other_track),
            )
            .unwrap();
        assert_eq!(
            ledger.begin(Track::Coding, &same_tenant_other_track),
            Ok(AddRetryState::Committed(verified_add(Track::Coding, &same_tenant_other_track)))
        );
    }

    #[test]
    fn payload_digest_is_track_bound() {
        let request = add_request("track-bound-digest");
        assert_ne!(
            add_payload_digest(Track::Textual, &request),
            add_payload_digest(Track::Coding, &request)
        );
    }

    #[test]
    fn idempotency_abandon_only_removes_matching_inflight_payload() {
        let request = add_request("write-4");
        let mut ledger = AddIdempotencyLedger::default();
        begin_textual(&mut ledger, &request).unwrap();
        assert_eq!(abandon_textual(&mut ledger, &request), Ok(()));
        assert_eq!(
            begin_textual(&mut ledger, &request),
            Ok(AddRetryState::New)
        );
    }

    #[test]
    fn idempotency_reservation_validates_request_before_claiming_request_id() {
        let mut invalid = add_request("");
        invalid.user_id = " ".into();
        let mut ledger = AddIdempotencyLedger::default();
        assert_eq!(
            begin_textual(&mut ledger, &invalid),
            Err(AmlReject::EmptyRequestId)
        );
        invalid.request_id = "valid-id".into();
        assert_eq!(
            begin_textual(&mut ledger, &invalid),
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
    fn unsupported_image_mime_and_malformed_base64_are_rejected() {
        let svg = AddRequest {
            request_id: "svg".into(),
            user_id: "user".into(),
            session_id: "session".into(),
            messages: vec![Message {
                role: Role::User,
                timestamp_ms: None,
                content: MessageContent::Parts(vec![
                    ContentPart::ImageDataUri("data:image/svg+xml;base64,PHN2Zz4=".into()),
                ]),
            }],
        };
        assert_eq!(
            validate_add_request(Track::Multimodal, &svg),
            Err(AmlReject::InvalidImageDataUri)
        );

        let bad_padding = AddRequest {
            request_id: "bad-padding".into(),
            user_id: "user".into(),
            session_id: "session".into(),
            messages: vec![Message {
                role: Role::User,
                timestamp_ms: None,
                content: MessageContent::Parts(vec![
                    ContentPart::ImageDataUri("data:image/png;base64,a===".into()),
                ]),
            }],
        };
        assert_eq!(
            validate_add_request(Track::Multimodal, &bad_padding),
            Err(AmlReject::InvalidBase64)
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
