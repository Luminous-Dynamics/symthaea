// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Minimal durable Add/Search service slice. This is a transparent baseline,
// not a production-ready AML submission. The local active row state is NOT a
// substitute for Mycelix rights/authority verification, erasure witnessing,
// independent qualification, or AML's frozen track preprocessing contract.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
};

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    middleware::{self, Next},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use uuid::Uuid;

const MAX_BODY_BYTES: usize = 48 * 1024 * 1024;
const MAX_MESSAGES: usize = 20;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_REQUEST_IMAGE_BYTES: usize = 30 * 1024 * 1024;
const MAX_SEARCH_RESPONSE_IMAGE_BYTES: usize = 30 * 1024 * 1024;
const MAX_TOP_K: usize = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Track { Textual, Coding, Multimodal }

impl Track {
    fn parse(value: &str) -> Result<Self, std::io::Error> {
        match value {
            "textual" => Ok(Self::Textual),
            "coding" => Ok(Self::Coding),
            "multimodal" => Ok(Self::Multimodal),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MACE_TRACK must be textual, coding, or multimodal",
            )),
        }
    }
    fn as_str(self) -> &'static str {
        match self { Self::Textual => "textual", Self::Coding => "coding", Self::Multimodal => "multimodal" }
    }
}

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
    api_key: Arc<Vec<u8>>,
    principal_id: String,
    tenant_id: String,
    track: Track,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AddRequest {
    request_id: String,
    messages: Vec<Message>,
    user_id: String,
    session_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    #[serde(default, deserialize_with = "deserialize_timestamp")]
    timestamp: Option<i64>,
    content: Content,
}

fn deserialize_timestamp<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    // Missing is handled by serde(default); explicit null is rejected.
    i64::deserialize(deserializer).map(Some)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Role { User, Assistant }

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum Content { Text(String), Parts(Vec<ContentPart>) }

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ImageUrl { url: String }

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    query: Content,
    #[serde(default)]
    options: Option<Vec<String>>,
    user_id: String,
    top_k: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AddResponse {
    success: bool,
    request_id: String,
    user_id: String,
    session_id: String,
}

#[derive(Clone, Debug, Serialize)]
struct SearchResponse { data: Vec<SearchItem> }

#[derive(Serialize)]
struct SearchTraceReceiptPayloadV1 {
    schema_version: &'static str,
    search_id: String,
    principal_id: String,
    tenant_id: String,
    user_scope_sha256: String,
    query_sha256: String,
    request_sha256: String,
    top_k: usize,
    ordered_memory_ids_sha256: String,
    response_sha256: String,
    retrieval_policy: &'static str,
    returned_at: String,
}

#[derive(Clone, Debug, Serialize)]
struct SearchItem {
    id: String,
    content: Content,
    score: f64,
    created_at: String,
}

#[derive(Debug)]
enum ApiError { BadRequest(&'static str), PayloadTooLarge, Unauthorized, Conflict, Unavailable, Internal }

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
            Self::PayloadTooLarge => (StatusCode::PAYLOAD_TOO_LARGE, "Search response image payload exceeds 30 MiB"),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "authentication required"),
            Self::Conflict => (StatusCode::UNPROCESSABLE_ENTITY, "request id is already bound to different content"),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "storage outcome unavailable; retry with the same request id"),
            Self::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "internal error"),
        };
        (status, Json(json!({"error": message}))).into_response()
    }
}

struct StoredCandidate { id: String, content: Content, searchable_text: String, created_at: String }

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn now_rfc3339() -> String { Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true) }

fn sha256_fields(fields: &[&str]) -> String {
    let mut bytes = Vec::new();
    for field in fields {
        bytes.extend_from_slice(&(field.len() as u64).to_be_bytes());
        bytes.extend_from_slice(field.as_bytes());
    }
    sha256_hex(&bytes)
}

fn stable_memory_id(
    state: &AppState,
    request_id: &str,
    payload_sha256: &str,
    message_ordinal: usize,
) -> String {
    let ordinal = message_ordinal.to_string();
    sha256_fields(&[
        "mace.memory-id.v1",
        &state.principal_id,
        &state.tenant_id,
        request_id,
        payload_sha256,
        &ordinal,
    ])
}

fn content_text(content: &Content) -> String {
    match content {
        Content::Text(text) => text.clone(),
        Content::Parts(parts) => parts.iter().filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            ContentPart::ImageUrl { .. } => None,
        }).collect::<Vec<_>>().join(" "),
    }
}

fn tokenize(value: &str) -> Vec<String> {
    let mut token = String::new();
    let mut output = Vec::new();
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() { token.push(ch); }
        else if !token.is_empty() { output.push(std::mem::take(&mut token)); }
    }
    if !token.is_empty() { output.push(token); }
    output
}

fn validate_image_data_uri(uri: &str) -> Result<usize, ApiError> {
    let (header, payload) = uri.split_once(',').ok_or(ApiError::BadRequest("invalid image data URI"))?;
    let expected = match header {
        "data:image/jpeg;base64" => "jpeg",
        "data:image/png;base64" => "png",
        "data:image/webp;base64" => "webp",
        _ => return Err(ApiError::BadRequest("only JPEG, PNG, and WebP data URIs are accepted")),
    };
    if payload.is_empty() || payload.len() > ((MAX_IMAGE_BYTES + 2) / 3) * 4 + 4 {
        return Err(ApiError::BadRequest("image payload is empty or exceeds the encoded size bound"));
    }
    let bytes = STANDARD.decode(payload).map_err(|_| ApiError::BadRequest("invalid Base64 image payload"))?;
    if bytes.len() > MAX_IMAGE_BYTES { return Err(ApiError::BadRequest("image exceeds the decoded 10 MiB limit")); }
    let valid_signature = match expected {
        "jpeg" => bytes.starts_with(&[0xff, 0xd8, 0xff]),
        "png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "webp" => bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP",
        _ => false,
    };
    if !valid_signature { return Err(ApiError::BadRequest("image bytes do not match the declared media type")); }
    Ok(bytes.len())
}

fn validate_content(track: Track, content: &Content) -> Result<usize, ApiError> {
    match content {
        Content::Text(value) => {
            if value.trim().is_empty() { return Err(ApiError::BadRequest("message content must not be empty")); }
            Ok(0)
        }
        Content::Parts(parts) => {
            if track != Track::Multimodal { return Err(ApiError::BadRequest("ordered content parts are only valid on the multimodal track")); }
            if parts.is_empty() { return Err(ApiError::BadRequest("content parts must not be empty")); }
            let mut image_bytes = 0usize;
            for part in parts {
                match part {
                    ContentPart::Text { text } if text.trim().is_empty() => return Err(ApiError::BadRequest("text content parts must not be empty")),
                    ContentPart::Text { .. } => {}
                    ContentPart::ImageUrl { image_url } => image_bytes = image_bytes.saturating_add(validate_image_data_uri(&image_url.url)?),
                }
            }
            Ok(image_bytes)
        }
    }
}

fn exceeds_wire_max_length(value: &str, max_chars: usize) -> bool {
    value.chars().count() > max_chars
}

fn validate_add(track: Track, request: &AddRequest) -> Result<(), ApiError> {
    if request.request_id.trim().is_empty() || exceeds_wire_max_length(&request.request_id, 4096) { return Err(ApiError::BadRequest("request_id must be non-empty and at most 4096 characters")); }
    if request.user_id.trim().is_empty() || exceeds_wire_max_length(&request.user_id, 4096) { return Err(ApiError::BadRequest("user_id must be non-empty and at most 4096 characters")); }
    if request.session_id.trim().is_empty() || exceeds_wire_max_length(&request.session_id, 4096) { return Err(ApiError::BadRequest("session_id must be non-empty and at most 4096 characters")); }
    if request.messages.is_empty() { return Err(ApiError::BadRequest("messages must not be empty")); }
    if track == Track::Textual && request.messages.len() > MAX_MESSAGES { return Err(ApiError::BadRequest("textual Add accepts at most 20 messages per chunk")); }
    let mut total_image_bytes = 0usize;
    for message in &request.messages {
        if message.timestamp.is_some_and(|timestamp| {
            timestamp < 0 || chrono::DateTime::<Utc>::from_timestamp_millis(timestamp).is_none()
        }) { return Err(ApiError::BadRequest("timestamp must be a representable non-negative Unix millisecond value")); }
        if track != Track::Multimodal && matches!(message.content, Content::Parts(_)) { return Err(ApiError::BadRequest("textual and coding tracks require string message content")); }
        total_image_bytes = total_image_bytes.saturating_add(validate_content(track, &message.content)?);
    }
    if total_image_bytes > MAX_REQUEST_IMAGE_BYTES { return Err(ApiError::BadRequest("aggregate decoded image bytes exceed 30 MiB")); }
    Ok(())
}

fn validate_search(track: Track, request: &SearchRequest) -> Result<(), ApiError> {
    if request.user_id.trim().is_empty() || exceeds_wire_max_length(&request.user_id, 4096) { return Err(ApiError::BadRequest("user_id must be non-empty and at most 4096 characters")); }
    if request.top_k == 0 || request.top_k > MAX_TOP_K { return Err(ApiError::BadRequest("top_k must be between 1 and 100")); }
    if request.options.as_ref().is_some_and(|values| values.iter().any(|value| value.trim().is_empty())) { return Err(ApiError::BadRequest("options must not contain empty strings")); }
    let image_bytes = validate_content(track, &request.query)?;
    if image_bytes > MAX_REQUEST_IMAGE_BYTES { return Err(ApiError::BadRequest("query image bytes exceed 30 MiB")); }
    Ok(())
}

fn validate_search_response_image_budget(
    track: Track,
    items: &[SearchItem],
    maximum_bytes: usize,
) -> Result<(), ApiError> {
    let mut total_image_bytes = 0usize;
    for item in items {
        total_image_bytes = total_image_bytes.saturating_add(validate_content(track, &item.content)?);
        if total_image_bytes > maximum_bytes {
            return Err(ApiError::PayloadTooLarge);
        }
    }
    Ok(())
}

fn memory_table_columns(connection: &Connection) -> Result<BTreeSet<String>, rusqlite::Error> {
    let mut statement = connection.prepare("PRAGMA table_info(memory_items)")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    let mut columns = BTreeSet::new();
    for row in rows {
        columns.insert(row?);
    }
    Ok(columns)
}

fn open_database(path: &Path) -> Result<Connection, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
    }
    let connection = Connection::open(path)?;
    connection.busy_timeout(std::time::Duration::from_secs(5))?;
    connection.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = FULL;
         PRAGMA foreign_keys = ON;
         CREATE TABLE IF NOT EXISTS memory_items (
             memory_id TEXT PRIMARY KEY,
             principal_id TEXT NOT NULL,
             tenant_id TEXT NOT NULL,
             user_id TEXT NOT NULL,
             session_id TEXT NOT NULL,
             request_id TEXT NOT NULL,
             role TEXT NOT NULL,
             content_json TEXT NOT NULL,
             searchable_text TEXT NOT NULL,
             event_timestamp_ms INTEGER,
             ingested_at TEXT NOT NULL,
             lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN ('active','revoked','erased'))
         );
         CREATE INDEX IF NOT EXISTS memory_scope_idx
             ON memory_items(principal_id, tenant_id, user_id, lifecycle_state);
         CREATE TABLE IF NOT EXISTS add_idempotency (
             principal_id TEXT NOT NULL,
             tenant_id TEXT NOT NULL,
             request_id TEXT NOT NULL,
             user_id TEXT NOT NULL,
             payload_sha256 TEXT NOT NULL,
             response_json TEXT NOT NULL,
             committed_at TEXT NOT NULL,
             PRIMARY KEY (principal_id, tenant_id, request_id)
         );
         CREATE TABLE IF NOT EXISTS search_trace_receipts (
             search_id TEXT PRIMARY KEY,
             principal_id TEXT NOT NULL,
             tenant_id TEXT NOT NULL,
             user_scope_sha256 TEXT NOT NULL,
             query_sha256 TEXT NOT NULL,
             request_sha256 TEXT NOT NULL,
             ordered_memory_ids_json TEXT NOT NULL,
             ordered_memory_ids_sha256 TEXT NOT NULL,
             response_sha256 TEXT NOT NULL,
             retrieval_policy TEXT NOT NULL,
             returned_at TEXT NOT NULL,
             receipt_json TEXT NOT NULL,
             receipt_sha256 TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS search_trace_scope_idx
             ON search_trace_receipts(principal_id, tenant_id, user_scope_sha256, returned_at);",
    )?;

    // Non-destructive v1 -> v2 migration. The first prototype stored ingestion
    // time in the created_at column; v2 distinguishes event time from ingest.
    // Rename the old column so existing local databases remain usable.
    let mut columns = memory_table_columns(&connection)?;
    if columns.contains("created_at") && !columns.contains("ingested_at") {
        connection.execute_batch(
            "ALTER TABLE memory_items RENAME COLUMN created_at TO ingested_at;"
        )?;
        columns.remove("created_at");
        columns.insert("ingested_at".to_owned());
    }
    if !columns.contains("event_timestamp_ms") {
        connection.execute_batch(
            "ALTER TABLE memory_items ADD COLUMN event_timestamp_ms INTEGER;"
        )?;
    }
    if !columns.contains("ingested_at") {
        connection.execute_batch(
            "ALTER TABLE memory_items ADD COLUMN ingested_at TEXT NOT NULL DEFAULT '';"
        )?;
    }
    connection.execute_batch("PRAGMA user_version = 2;")?;

    Ok(connection)
}

fn add_transaction(state: &AppState, request: AddRequest) -> Result<AddResponse, ApiError> {
    validate_add(state.track, &request)?;
    // Serde's struct-field order and preserved message/part order define the
    // canonical request bytes for this version of the adapter.
    let canonical = serde_json::to_vec(&request).map_err(|_| ApiError::Internal)?;
    let payload_digest = sha256_hex(&canonical);
    let mut connection = state.db.lock().map_err(|_| ApiError::Unavailable)?;
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|_| ApiError::Unavailable)?;

    let existing: Option<(String, String)> = tx.query_row(
        "SELECT payload_sha256, response_json FROM add_idempotency
         WHERE principal_id = ?1 AND tenant_id = ?2 AND request_id = ?3",
        params![state.principal_id, state.tenant_id, request.request_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).optional().map_err(|_| ApiError::Unavailable)?;

    if let Some((stored_digest, response_json)) = existing {
        if stored_digest != payload_digest { return Err(ApiError::Conflict); }
        return serde_json::from_str(&response_json).map_err(|_| ApiError::Internal);
    }

    let response = AddResponse {
        success: true,
        request_id: request.request_id.clone(),
        user_id: request.user_id.clone(),
        session_id: request.session_id.clone(),
    };
    let committed_at = now_rfc3339();

    for (message_ordinal, message) in request.messages.iter().enumerate() {
        let memory_id = stable_memory_id(
            state,
            &request.request_id,
            &payload_digest,
            message_ordinal,
        );
        let role = match message.role { Role::User => "user", Role::Assistant => "assistant" };
        let content_json = serde_json::to_string(&message.content).map_err(|_| ApiError::Internal)?;
        let searchable_text = content_text(&message.content);
        tx.execute(
            "INSERT INTO memory_items (
                 memory_id, principal_id, tenant_id, user_id, session_id, request_id,
                 role, content_json, searchable_text, event_timestamp_ms, ingested_at, lifecycle_state
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'active')",
            params![
                memory_id, state.principal_id, state.tenant_id, request.user_id,
                request.session_id, request.request_id, role, content_json,
                searchable_text, message.timestamp, committed_at
            ],
        ).map_err(|_| ApiError::Unavailable)?;
    }

    let response_json = serde_json::to_string(&response).map_err(|_| ApiError::Internal)?;
    tx.execute(
        "INSERT INTO add_idempotency (
             principal_id, tenant_id, request_id, user_id, payload_sha256, response_json, committed_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            state.principal_id, state.tenant_id, request.request_id, request.user_id,
            payload_digest, response_json, committed_at
        ],
    ).map_err(|_| ApiError::Unavailable)?;

    // Source rows, searchable projection and the idempotency receipt commit
    // in one FULL-synchronous transaction. A lost response is reconciled by
    // retrying the same request ID and identical canonical payload.
    tx.commit().map_err(|_| ApiError::Unavailable)?;
    Ok(response)
}

fn bm25_score(query_terms: &[String], doc_terms: &[String], df: &BTreeMap<String, usize>, n_docs: usize, avg_len: f64) -> f64 {
    if query_terms.is_empty() || doc_terms.is_empty() || n_docs == 0 { return 0.0; }
    let mut tf = BTreeMap::<&str, usize>::new();
    for term in doc_terms { *tf.entry(term.as_str()).or_default() += 1; }
    let doc_len = doc_terms.len() as f64;
    let k1 = 1.2;
    let b = 0.75;
    let mut score = 0.0;
    for term in query_terms {
        let frequency = *tf.get(term.as_str()).unwrap_or(&0) as f64;
        if frequency == 0.0 { continue; }
        let document_frequency = *df.get(term).unwrap_or(&0) as f64;
        let idf = ((n_docs as f64 - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).ln();
        let denominator = frequency + k1 * (1.0 - b + b * doc_len / avg_len.max(1.0));
        score += idf * frequency * (k1 + 1.0) / denominator;
    }
    score
}

fn search_query(state: &AppState, request: SearchRequest) -> Result<SearchResponse, ApiError> {
    validate_search(state.track, &request)?;
    let query_terms = tokenize(&content_text(&request.query));

    // Search exposure and its internal receipt share one transaction. If the
    // receipt cannot be committed, no evidence is returned to the caller.
    let mut connection = state.db.lock().map_err(|_| ApiError::Unavailable)?;
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| ApiError::Unavailable)?;
    let mut statement = tx.prepare(
        "SELECT memory_id, content_json, searchable_text, event_timestamp_ms, ingested_at
         FROM memory_items
         WHERE principal_id = ?1 AND tenant_id = ?2 AND user_id = ?3
           AND lifecycle_state = 'active'",
    ).map_err(|_| ApiError::Unavailable)?;
    let rows = statement.query_map(params![state.principal_id, state.tenant_id, request.user_id], |row| {
        let content_json: String = row.get(1)?;
        let content: Content = serde_json::from_str(&content_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(error))
        })?;
        let event_timestamp_ms: Option<i64> = row.get(3)?;
        let ingested_at: String = row.get(4)?;
        let created_at = event_timestamp_ms
            .and_then(chrono::DateTime::<Utc>::from_timestamp_millis)
            .map(|timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Millis, true))
            .unwrap_or(ingested_at);
        Ok(StoredCandidate { id: row.get(0)?, content, searchable_text: row.get(2)?, created_at })
    }).map_err(|_| ApiError::Unavailable)?;

    let mut candidates = Vec::new();
    for row in rows { candidates.push(row.map_err(|_| ApiError::Unavailable)?); }
    drop(statement);

    let doc_terms: Vec<Vec<String>> = candidates.iter().map(|item| tokenize(&item.searchable_text)).collect();
    let mut df = BTreeMap::<String, usize>::new();
    let mut total_len = 0usize;
    for terms in &doc_terms {
        total_len += terms.len();
        let unique: BTreeSet<&str> = terms.iter().map(String::as_str).collect();
        for term in unique { *df.entry(term.to_owned()).or_default() += 1; }
    }
    let n_docs = candidates.len();
    let avg_len = if n_docs == 0 { 1.0 } else { total_len as f64 / n_docs as f64 };

    let mut ranked: Vec<(f64, StoredCandidate)> = candidates.into_iter().zip(doc_terms)
        .map(|(candidate, terms)| (bm25_score(&query_terms, &terms, &df, n_docs, avg_len), candidate))
        .filter(|(score, _)| *score > 0.0)
        .collect();

    // Deterministic ties: descending score, then ascending immutable memory ID.
    ranked.sort_by(|(score_a, a), (score_b, b)| score_b.total_cmp(score_a).then_with(|| a.id.cmp(&b.id)));

    let data = ranked.into_iter().take(request.top_k).map(|(score, candidate)| SearchItem {
        id: candidate.id, content: candidate.content, score, created_at: candidate.created_at,
    }).collect::<Vec<_>>();
    validate_search_response_image_budget(
        state.track,
        &data,
        MAX_SEARCH_RESPONSE_IMAGE_BYTES,
    )?;
    let response = SearchResponse { data };

    // Persist only digests and opaque memory IDs: never store raw query text or
    // raw user identifiers in this retrieval audit stream.
    let search_id = Uuid::new_v4().to_string();
    let returned_at = now_rfc3339();
    let user_scope_sha256 = sha256_fields(&[
        &state.principal_id,
        &state.tenant_id,
        &request.user_id,
    ]);
    let query_bytes = serde_json::to_vec(&request.query).map_err(|_| ApiError::Internal)?;
    let request_bytes = serde_json::to_vec(&request).map_err(|_| ApiError::Internal)?;
    let ordered_memory_ids: Vec<&str> = response.data.iter().map(|item| item.id.as_str()).collect();
    let ordered_memory_ids_json = serde_json::to_string(&ordered_memory_ids).map_err(|_| ApiError::Internal)?;
    let ordered_memory_ids_sha256 = sha256_hex(ordered_memory_ids_json.as_bytes());
    let response_bytes = serde_json::to_vec(&response).map_err(|_| ApiError::Internal)?;
    let response_sha256 = sha256_hex(&response_bytes);
    let receipt_payload = SearchTraceReceiptPayloadV1 {
        schema_version: "mace.search-trace-receipt.v1",
        search_id: search_id.clone(),
        principal_id: state.principal_id.clone(),
        tenant_id: state.tenant_id.clone(),
        user_scope_sha256: user_scope_sha256.clone(),
        query_sha256: sha256_hex(&query_bytes),
        request_sha256: sha256_hex(&request_bytes),
        top_k: request.top_k,
        ordered_memory_ids_sha256: ordered_memory_ids_sha256.clone(),
        response_sha256: response_sha256.clone(),
        retrieval_policy: "bm25_unicode_token_v1",
        returned_at: returned_at.clone(),
    };
    let receipt_json = serde_json::to_string(&receipt_payload).map_err(|_| ApiError::Internal)?;
    let receipt_sha256 = sha256_hex(receipt_json.as_bytes());
    tx.execute(
        "INSERT INTO search_trace_receipts (
             search_id, principal_id, tenant_id, user_scope_sha256, query_sha256,
             request_sha256, ordered_memory_ids_json, ordered_memory_ids_sha256,
             response_sha256, retrieval_policy, returned_at, receipt_json, receipt_sha256
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            search_id, state.principal_id, state.tenant_id, user_scope_sha256,
            receipt_payload.query_sha256, receipt_payload.request_sha256,
            ordered_memory_ids_json, ordered_memory_ids_sha256, response_sha256,
            receipt_payload.retrieval_policy, returned_at, receipt_json, receipt_sha256
        ],
    ).map_err(|_| ApiError::Unavailable)?;
    tx.commit().map_err(|_| ApiError::Unavailable)?;
    Ok(response)
}

fn presented_key(headers: &HeaderMap) -> Option<Vec<u8>> {
    if let Some(value) = headers.get(header::AUTHORIZATION).and_then(|value| value.to_str().ok()) {
        for prefix in ["Bearer ", "Token "] {
            if let Some(token) = value.strip_prefix(prefix) { return Some(token.as_bytes().to_vec()); }
        }
    }
    headers.get("x-api-key").and_then(|value| value.to_str().ok()).map(|value| value.as_bytes().to_vec())
}

fn authenticate(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
    let presented = presented_key(headers).ok_or(ApiError::Unauthorized)?;
    let expected = state.api_key.as_slice();
    if presented.len() != expected.len() || !bool::from(presented.as_slice().ct_eq(expected)) {
        return Err(ApiError::Unauthorized);
    }
    Ok(())
}

async fn authenticate_request(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    authenticate(request.headers(), &state)?;
    Ok(next.run(request).await)
}

async fn health(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let ready = tokio::task::spawn_blocking(move || {
        let connection = state.db.lock().map_err(|_| ApiError::Unavailable)?;
        let value: i64 = connection.query_row("SELECT 1", [], |row| row.get(0))
            .map_err(|_| ApiError::Unavailable)?;
        Ok::<bool, ApiError>(value == 1)
    }).await.map_err(|_| ApiError::Internal)??;
    if !ready { return Err(ApiError::Unavailable); }
    Ok((StatusCode::OK, Json(json!({"status":"ready","service":"mace-aml-hosted-service-v0.1"}))))
}

async fn add(State(state): State<Arc<AppState>>, Json(request): Json<AddRequest>)
    -> Result<Json<AddResponse>, ApiError>
{
    let result = tokio::task::spawn_blocking(move || add_transaction(&state, request))
        .await.map_err(|_| ApiError::Internal)??;
    Ok(Json(result))
}

async fn search(State(state): State<Arc<AppState>>, Json(request): Json<SearchRequest>)
    -> Result<Json<SearchResponse>, ApiError>
{
    let result = tokio::task::spawn_blocking(move || search_query(&state, request))
        .await.map_err(|_| ApiError::Internal)??;
    Ok(Json(result))
}

fn required_env(name: &str) -> Result<String, std::io::Error> {
    let value = env::var(name).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::NotFound, format!("{name} must be configured"))
    })?;
    if value.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{name} must not be empty"),
        ));
    }
    Ok(value)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key = required_env("MACE_API_KEY")?;
    let principal_id = required_env("MACE_PRINCIPAL_ID")?;
    let tenant_id = required_env("MACE_TENANT_ID")?;
    let track = Track::parse(&env::var("MACE_TRACK").unwrap_or_else(|_| "textual".into()))?;
    let database_path = env::var("MACE_DB_PATH").unwrap_or_else(|_| "state/mace-aml.sqlite3".into());
    let bind_address: SocketAddr = env::var("MACE_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into()).parse()?;

    let connection = open_database(Path::new(&database_path))?;
    let state = Arc::new(AppState {
        db: Arc::new(Mutex::new(connection)),
        api_key: Arc::new(api_key.into_bytes()),
        principal_id, tenant_id, track,
    });

    let protected_routes = Router::new()
        .route("/add", post(add))
        .route("/search", post(search))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate_request));

    let app = Router::new()
        .route("/health", get(health))
        .merge(protected_routes)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);

    let listener = TcpListener::bind(bind_address).await?;
    eprintln!("mace-aml-hosted-service listening on {} (track={})", listener.local_addr()?, track.as_str());
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> AppState {
        let connection = open_database(Path::new(":memory:")).unwrap();
        AppState {
            db: Arc::new(Mutex::new(connection)),
            api_key: Arc::new(b"test-secret".to_vec()),
            principal_id: "participant-1".into(),
            tenant_id: "tenant-1".into(),
            track: Track::Textual,
        }
    }

    fn add_request(id: &str, text: &str) -> AddRequest {
        AddRequest {
            request_id: id.into(), user_id: "user-1".into(), session_id: "session-1".into(),
            messages: vec![Message { role: Role::User, timestamp: Some(1_795_492_800_000), content: Content::Text(text.into()) }],
        }
    }

    #[test]
    fn add_is_immediately_searchable_from_the_same_store() {
        let state = test_state();
        let response = add_transaction(&state, add_request("r-1", "The red bicycle is stored in the garage")).unwrap();
        assert!(response.success);
        let results = search_query(&state, SearchRequest {
            query: Content::Text("red bicycle".into()), options: None, user_id: "user-1".into(), top_k: 10,
        }).unwrap();
        assert_eq!(results.data.len(), 1);
        assert_eq!(content_text(&results.data[0].content), "The red bicycle is stored in the garage");
    }

    #[test]
    fn same_logical_add_gets_same_memory_ids_across_fresh_stores() {
        let first_state = test_state();
        let second_state = test_state();
        let request = add_request("stable-id", "same logical source record");
        add_transaction(&first_state, request.clone()).unwrap();
        add_transaction(&second_state, request).unwrap();

        let query = SearchRequest {
            query: Content::Text("logical source record".into()),
            options: None,
            user_id: "user-1".into(),
            top_k: 5,
        };
        let first = search_query(&first_state, query.clone()).unwrap();
        let second = search_query(&second_state, query).unwrap();
        assert_eq!(first.data.len(), 1);
        assert_eq!(second.data.len(), 1);
        assert_eq!(first.data[0].id, second.data[0].id);
    }

    #[test]
    fn identical_retry_does_not_duplicate_rows() {
        let state = test_state();
        let request = add_request("r-2", "idempotent durable memory");
        let first = add_transaction(&state, request.clone()).unwrap();
        let second = add_transaction(&state, request).unwrap();
        assert_eq!(first.request_id, second.request_id);
        let connection = state.db.lock().unwrap();
        let count: i64 = connection.query_row("SELECT COUNT(*) FROM memory_items", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn request_id_rebinding_to_changed_content_is_rejected() {
        let state = test_state();
        add_transaction(&state, add_request("r-3", "original content")).unwrap();
        assert!(matches!(add_transaction(&state, add_request("r-3", "changed content")), Err(ApiError::Conflict)));
        let connection = state.db.lock().unwrap();
        let count: i64 = connection.query_row("SELECT COUNT(*) FROM memory_items", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn user_scope_isolation_prevents_cross_user_retrieval() {
        let state = test_state();
        add_transaction(&state, add_request("r-4", "private blue notebook")).unwrap();
        let results = search_query(&state, SearchRequest {
            query: Content::Text("blue notebook".into()), options: None, user_id: "user-2".into(), top_k: 10,
        }).unwrap();
        assert!(results.data.is_empty());
    }

    #[test]
    fn legacy_created_at_database_migrates_without_losing_rows() {
        let path = std::env::temp_dir().join(format!("mace-legacy-{}.sqlite3", Uuid::new_v4()));
        {
            let legacy = Connection::open(&path).unwrap();
            legacy.execute_batch(
                "CREATE TABLE memory_items (
                    memory_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL
                 );"
            ).unwrap();
            legacy.execute(
                "INSERT INTO memory_items VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    "legacy-memory", "participant-1", "tenant-1", "user-1",
                    "session-1", "legacy-request", "user", "\"legacy source\"",
                    "legacy source", "2026-11-23T10:00:00Z", "active"
                ],
            ).unwrap();
        }
        let migrated = open_database(&path).unwrap();
        let columns = memory_table_columns(&migrated).unwrap();
        assert!(columns.contains("event_timestamp_ms"));
        assert!(columns.contains("ingested_at"));
        assert!(!columns.contains("created_at"));
        let (content, ingested_at): (String, String) = migrated.query_row(
            "SELECT content_json, ingested_at FROM memory_items WHERE memory_id = 'legacy-memory'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).unwrap();
        assert_eq!(content, "\"legacy source\"");
        assert_eq!(ingested_at, "2026-11-23T10:00:00Z");
        drop(migrated);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite3-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite3-shm"));
    }

    #[test]
    fn file_backed_store_survives_process_style_reopen() {
        let path = std::env::temp_dir().join(format!("mace-hosted-{}.sqlite3", Uuid::new_v4()));
        let state = AppState {
            db: Arc::new(Mutex::new(open_database(&path).unwrap())),
            api_key: Arc::new(b"test-secret".to_vec()),
            principal_id: "participant-1".into(),
            tenant_id: "tenant-1".into(),
            track: Track::Textual,
        };
        add_transaction(&state, add_request("r-persist", "durable memory survives restart")).unwrap();
        drop(state);

        let reopened = AppState {
            db: Arc::new(Mutex::new(open_database(&path).unwrap())),
            api_key: Arc::new(b"test-secret".to_vec()),
            principal_id: "participant-1".into(),
            tenant_id: "tenant-1".into(),
            track: Track::Textual,
        };
        let results = search_query(&reopened, SearchRequest {
            query: Content::Text("survives restart".into()),
            options: None,
            user_id: "user-1".into(),
            top_k: 5,
        }).unwrap();
        assert_eq!(results.data.len(), 1);
        drop(reopened);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite3-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite3-shm"));
    }

    #[test]
    fn concurrent_identical_retries_commit_only_once() {
        let state = test_state();
        let request = add_request("r-concurrent", "concurrent idempotency receipt");
        let a_state = state.clone();
        let b_state = state.clone();
        let a_request = request.clone();
        let a = std::thread::spawn(move || add_transaction(&a_state, a_request).unwrap());
        let b = std::thread::spawn(move || add_transaction(&b_state, request).unwrap());
        let first = a.join().unwrap();
        let second = b.join().unwrap();
        assert_eq!(first.request_id, second.request_id);
        let connection = state.db.lock().unwrap();
        let count: i64 = connection.query_row("SELECT COUNT(*) FROM memory_items", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn missing_optional_timestamp_is_accepted_but_explicit_null_is_rejected() {
        let missing = serde_json::json!({"role":"user","content":"hello"});
        assert!(serde_json::from_value::<Message>(missing).is_ok());
        let explicit_null = serde_json::json!({"role":"user","timestamp":null,"content":"hello"});
        assert!(serde_json::from_value::<Message>(explicit_null).is_err());
    }

    #[test]
    fn wire_max_length_counts_unicode_characters_not_utf8_bytes() {
        let valid = "é".repeat(4096);
        let too_long = "é".repeat(4097);
        assert!(!exceeds_wire_max_length(&valid, 4096));
        assert!(exceeds_wire_max_length(&too_long, 4096));
        assert!(valid.len() > 4096);
    }

    #[test]
    fn every_search_commits_a_trace_receipt_before_returning_results() {
        let state = test_state();
        add_transaction(&state, add_request("r-audit", "red bicycle in the garage")).unwrap();
        let response = search_query(&state, SearchRequest {
            query: Content::Text("red bicycle".into()),
            options: None,
            user_id: "private-user-scope".into(),
            top_k: 10,
        }).unwrap();
        assert_eq!(response.data.len(), 1);

        let connection = state.db.lock().unwrap();
        let (ids_json, ids_sha, response_sha, receipt_json, receipt_sha, query_sha, scope_sha):
            (String, String, String, String, String, String, String) = connection.query_row(
                "SELECT ordered_memory_ids_json, ordered_memory_ids_sha256, response_sha256,
                        receipt_json, receipt_sha256, query_sha256, user_scope_sha256
                 FROM search_trace_receipts",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
            ).unwrap();
        assert_eq!(ids_json, serde_json::to_string(&vec![response.data[0].id.as_str()]).unwrap());
        assert_eq!(ids_sha, sha256_hex(ids_json.as_bytes()));
        assert_eq!(response_sha, sha256_hex(&serde_json::to_vec(&response).unwrap()));
        assert_eq!(receipt_sha, sha256_hex(receipt_json.as_bytes()));
        assert_eq!(query_sha, sha256_hex(&serde_json::to_vec(&Content::Text("red bicycle".into())).unwrap()));
        assert_eq!(scope_sha, sha256_fields(&["participant-1", "tenant-1", "private-user-scope"]));
        assert!(!receipt_json.contains("red bicycle"));
        assert!(!receipt_json.contains("private-user-scope"));
    }

    #[test]
    fn search_fails_closed_when_exposure_receipt_cannot_be_persisted() {
        let state = test_state();
        add_transaction(&state, add_request("r-audit-fail", "evidence must not escape without receipt")).unwrap();
        {
            let connection = state.db.lock().unwrap();
            connection.execute("DROP TABLE search_trace_receipts", []).unwrap();
        }
        let result = search_query(&state, SearchRequest {
            query: Content::Text("evidence must not escape".into()),
            options: None,
            user_id: "user-1".into(),
            top_k: 5,
        });
        assert!(matches!(result, Err(ApiError::Unavailable)));
    }

    #[test]
    fn multimodal_search_response_enforces_aggregate_image_budget_without_truncation() {
        let image = Content::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl { url: "data:image/png;base64,iVBORw0KGgo=".into() },
        }]);
        let items = vec![
            SearchItem { id: "m1".into(), content: image.clone(), score: 1.0, created_at: "2026-11-24T00:00:00Z".into() },
            SearchItem { id: "m2".into(), content: image, score: 0.5, created_at: "2026-11-24T00:00:01Z".into() },
        ];
        assert!(validate_search_response_image_budget(Track::Multimodal, &items, 16).is_ok());
        assert!(matches!(
            validate_search_response_image_budget(Track::Multimodal, &items, 15),
            Err(ApiError::PayloadTooLarge)
        ));
        assert_eq!(ApiError::PayloadTooLarge.into_response().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn empty_search_is_also_audited() {
        let state = test_state();
        let response = search_query(&state, SearchRequest {
            query: Content::Text("no matching term".into()),
            options: None,
            user_id: "user-1".into(),
            top_k: 5,
        }).unwrap();
        assert!(response.data.is_empty());
        let connection = state.db.lock().unwrap();
        let count: i64 = connection.query_row("SELECT COUNT(*) FROM search_trace_receipts", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn invalid_top_k_is_rejected() {
        let state = test_state();
        let request = SearchRequest { query: Content::Text("anything".into()), options: None, user_id: "user-1".into(), top_k: 0 };
        assert!(matches!(search_query(&state, request), Err(ApiError::BadRequest(_))));
    }

    #[test]
    fn payload_conflict_uses_permanent_422_response() {
        let response = ApiError::Conflict.into_response();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn multimodal_image_bytes_must_match_declared_signature() {
        let invalid = "data:image/png;base64,aGVsbG8=";
        assert!(matches!(validate_image_data_uri(invalid), Err(ApiError::BadRequest(_))));
    }
}
