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
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use uuid::Uuid;

const MAX_BODY_BYTES: usize = 48 * 1024 * 1024;
const MAX_MESSAGES: usize = 20;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_REQUEST_IMAGE_BYTES: usize = 30 * 1024 * 1024;
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
    #[serde(default)]
    timestamp: Option<i64>,
    content: Content,
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

#[derive(Clone, Debug, Deserialize)]
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

#[derive(Clone, Debug, Serialize)]
struct SearchItem {
    id: String,
    content: Content,
    score: f64,
    created_at: String,
}

#[derive(Debug)]
enum ApiError { BadRequest(&'static str), Unauthorized, Conflict, Unavailable, Internal }

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
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

fn validate_add(track: Track, request: &AddRequest) -> Result<(), ApiError> {
    if request.request_id.trim().is_empty() || request.request_id.len() > 4096 { return Err(ApiError::BadRequest("request_id must be non-empty and at most 4096 bytes")); }
    if request.user_id.trim().is_empty() || request.user_id.len() > 4096 { return Err(ApiError::BadRequest("user_id must be non-empty and at most 4096 bytes")); }
    if request.session_id.trim().is_empty() || request.session_id.len() > 4096 { return Err(ApiError::BadRequest("session_id must be non-empty and at most 4096 bytes")); }
    if request.messages.is_empty() { return Err(ApiError::BadRequest("messages must not be empty")); }
    if track == Track::Textual && request.messages.len() > MAX_MESSAGES { return Err(ApiError::BadRequest("textual Add accepts at most 20 messages per chunk")); }
    let mut total_image_bytes = 0usize;
    for message in &request.messages {
        if message.timestamp.is_some_and(|timestamp| timestamp < 0) { return Err(ApiError::BadRequest("timestamp must be a non-negative Unix millisecond value")); }
        if track != Track::Multimodal && matches!(message.content, Content::Parts(_)) { return Err(ApiError::BadRequest("textual and coding tracks require string message content")); }
        total_image_bytes = total_image_bytes.saturating_add(validate_content(track, &message.content)?);
    }
    if total_image_bytes > MAX_REQUEST_IMAGE_BYTES { return Err(ApiError::BadRequest("aggregate decoded image bytes exceed 30 MiB")); }
    Ok(())
}

fn validate_search(track: Track, request: &SearchRequest) -> Result<(), ApiError> {
    if request.user_id.trim().is_empty() || request.user_id.len() > 4096 { return Err(ApiError::BadRequest("user_id must be non-empty and at most 4096 bytes")); }
    if request.top_k == 0 || request.top_k > MAX_TOP_K { return Err(ApiError::BadRequest("top_k must be between 1 and 100")); }
    if request.options.as_ref().is_some_and(|values| values.iter().any(|value| value.trim().is_empty())) { return Err(ApiError::BadRequest("options must not contain empty strings")); }
    let image_bytes = validate_content(track, &request.query)?;
    if image_bytes > MAX_REQUEST_IMAGE_BYTES { return Err(ApiError::BadRequest("query image bytes exceed 30 MiB")); }
    Ok(())
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
             created_at TEXT NOT NULL,
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
         );",
    )?;
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

    for message in &request.messages {
        let memory_id = Uuid::new_v4().to_string();
        let role = match message.role { Role::User => "user", Role::Assistant => "assistant" };
        let content_json = serde_json::to_string(&message.content).map_err(|_| ApiError::Internal)?;
        let searchable_text = content_text(&message.content);
        tx.execute(
            "INSERT INTO memory_items (
                 memory_id, principal_id, tenant_id, user_id, session_id, request_id,
                 role, content_json, searchable_text, created_at, lifecycle_state
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'active')",
            params![
                memory_id, state.principal_id, state.tenant_id, request.user_id,
                request.session_id, request.request_id, role, content_json,
                searchable_text, committed_at
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
    if query_terms.is_empty() { return Ok(SearchResponse { data: Vec::new() }); }

    let connection = state.db.lock().map_err(|_| ApiError::Unavailable)?;
    let mut statement = connection.prepare(
        "SELECT memory_id, content_json, searchable_text, created_at
         FROM memory_items
         WHERE principal_id = ?1 AND tenant_id = ?2 AND user_id = ?3
           AND lifecycle_state = 'active'",
    ).map_err(|_| ApiError::Unavailable)?;
    let rows = statement.query_map(params![state.principal_id, state.tenant_id, request.user_id], |row| {
        let content_json: String = row.get(1)?;
        let content: Content = serde_json::from_str(&content_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(error))
        })?;
        Ok(StoredCandidate { id: row.get(0)?, content, searchable_text: row.get(2)?, created_at: row.get(3)? })
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
    }).collect();
    Ok(SearchResponse { data })
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

async fn add(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(request): Json<AddRequest>)
    -> Result<Json<AddResponse>, ApiError>
{
    authenticate(&headers, &state)?;
    let result = tokio::task::spawn_blocking(move || add_transaction(&state, request))
        .await.map_err(|_| ApiError::Internal)??;
    Ok(Json(result))
}

async fn search(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(request): Json<SearchRequest>)
    -> Result<Json<SearchResponse>, ApiError>
{
    authenticate(&headers, &state)?;
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

    let app = Router::new()
        .route("/health", get(health))
        .route("/add", post(add))
        .route("/search", post(search))
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
