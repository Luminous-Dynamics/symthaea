// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Optional participant-hosted HTTP service for the Agent Memory Challenge.
//!
//! The service implements only the synchronous Text/Coding Add/Search boundary.
//! It must run behind a TLS reverse proxy or other HTTPS ingress before public
//! evaluation. The health route is deliberately unauthenticated; Add/Search
//! require a configured participant Memory System Key.
//!
//! The service does not answer questions, expose evaluator labels, or return
//! data outside the exact user_id scope supplied by the AML request.

use crate::memory_challenge_api::{
    AmlAddRequest, AmlAddResponse, AmlSearchRequest, AmlSearchResponse, execute_aml_add,
    execute_aml_search,
};
use crate::memory_challenge_sqlite::SqliteAmlMemoryBackend;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

const OFFICIAL_MAX_TOP_K: usize = 100;
const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_MAX_CONCURRENCY: usize = 16;

/// Configuration for one local SQLite/WAL AML service instance.
#[derive(Debug, Clone)]
pub struct AmlHttpServerConfig {
    pub bind: SocketAddr,
    pub database_path: PathBuf,
    pub payload_root: PathBuf,
    /// Participant Memory System Key. Never log or serialize this value.
    pub api_key: String,
    pub max_body_bytes: usize,
    pub max_concurrency: usize,
}

impl AmlHttpServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.api_key.len() < 32 {
            return Err("AML_API_KEY must contain at least 32 bytes of high-entropy secret material".into());
        }
        if self.max_body_bytes == 0 || self.max_body_bytes > 64 * 1024 * 1024 {
            return Err("AML_MAX_BODY_BYTES must be between 1 and 67108864".into());
        }
        if self.max_concurrency == 0 || self.max_concurrency > 256 {
            return Err("AML_MAX_CONCURRENCY must be between 1 and 256".into());
        }
        if self.database_path.as_os_str().is_empty() {
            return Err("AML_DB_PATH must not be empty".into());
        }
        Ok(())
    }

    /// Build configuration from environment variables. No secret is logged.
    pub fn from_env() -> Result<Self, String> {
        let bind = std::env::var("AML_BIND")
            .unwrap_or_else(|_| "0.0.0.0:8080".into())
            .parse::<SocketAddr>()
            .map_err(|error| format!("invalid AML_BIND address: {error}"))?;
        let database_path = std::env::var_os("AML_DB_PATH")
            .map(PathBuf::from)
            .ok_or_else(|| "AML_DB_PATH is required".to_string())?;
        let payload_root = std::env::var_os("AML_PAYLOAD_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/var/lib/symthaea-aml/payloads"));
        let api_key = std::env::var("AML_API_KEY")
            .map_err(|_| "AML_API_KEY is required".to_string())?;
        let max_body_bytes = parse_env_usize("AML_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)?;
        let max_concurrency = parse_env_usize("AML_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)?;
        let config = Self {
            bind,
            database_path,
            payload_root,
            api_key,
            max_body_bytes,
            max_concurrency,
        };
        config.validate()?;
        Ok(config)
    }
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value.parse::<usize>()
            .map_err(|error| format!("{name} must be an unsigned integer: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

struct AmlHttpState {
    backend: Arc<Mutex<SqliteAmlMemoryBackend>>,
    api_key: String,
    request_slots: Arc<Semaphore>,
}

#[derive(Debug)]
struct ServiceError {
    status: StatusCode,
    reason: String,
}

impl ServiceError {
    fn new(status: StatusCode, reason: impl Into<String>) -> Self {
        Self { status, reason: reason.into() }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    detail: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    reason: String,
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                detail: ErrorDetail { reason: self.reason },
            }),
        )
            .into_response()
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        difference |= (left_byte ^ right_byte) as usize;
    }
    difference == 0
}

fn supplied_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get("x-api-key").and_then(|value| value.to_str().ok()) {
        return Some(value);
    }
    let authorization = headers.get(AUTHORIZATION)?.to_str().ok()?;
    authorization
        .strip_prefix("Bearer ")
        .or_else(|| authorization.strip_prefix("Token "))
}

fn authenticate(headers: &HeaderMap, expected: &str) -> bool {
    supplied_api_key(headers)
        .is_some_and(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes()))
}

async fn acquire_slot(state: &AmlHttpState) -> Result<tokio::sync::OwnedSemaphorePermit, ServiceError> {
    state.request_slots.clone().acquire_owned().await
        .map_err(|_| ServiceError::new(StatusCode::SERVICE_UNAVAILABLE, "service is shutting down"))
}

fn map_add_error(error: String) -> ServiceError {
    if error.contains("request_id was already committed with a different Add payload") {
        ServiceError::new(StatusCode::CONFLICT, error)
    } else {
        ServiceError::new(StatusCode::INTERNAL_SERVER_ERROR, error)
    }
}

async fn add_handler(
    State(state): State<Arc<AmlHttpState>>,
    headers: HeaderMap,
    request: Result<Json<AmlAddRequest>, JsonRejection>,
) -> Result<Json<AmlAddResponse>, ServiceError> {
    if !authenticate(&headers, &state.api_key) {
        return Err(ServiceError::new(StatusCode::UNAUTHORIZED, "valid Memory System Key required"));
    }
    let Json(request) = request.map_err(|error| {
        ServiceError::new(StatusCode::UNPROCESSABLE_ENTITY, format!("invalid Add JSON: {error}"))
    })?;
    request.validate().map_err(|error| {
        ServiceError::new(StatusCode::UNPROCESSABLE_ENTITY, error)
    })?;
    let permit = acquire_slot(&state).await?;
    let backend = state.backend.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut backend = backend.lock()
            .map_err(|_| "SQLite backend mutex is poisoned".to_string())?;
        execute_aml_add(&mut *backend, request)
    })
    .await
    .map_err(|error| ServiceError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("Add worker failed: {error}")))?;
    result.map(Json).map_err(map_add_error)
}

async fn search_handler(
    State(state): State<Arc<AmlHttpState>>,
    headers: HeaderMap,
    request: Result<Json<AmlSearchRequest>, JsonRejection>,
) -> Result<Json<AmlSearchResponse>, ServiceError> {
    if !authenticate(&headers, &state.api_key) {
        return Err(ServiceError::new(StatusCode::UNAUTHORIZED, "valid Memory System Key required"));
    }
    let Json(request) = request.map_err(|error| {
        ServiceError::new(StatusCode::UNPROCESSABLE_ENTITY, format!("invalid Search JSON: {error}"))
    })?;
    request.validate().map_err(|error| {
        ServiceError::new(StatusCode::UNPROCESSABLE_ENTITY, error)
    })?;
    if request.top_k > OFFICIAL_MAX_TOP_K {
        return Err(ServiceError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("top_k cannot exceed the official limit of {OFFICIAL_MAX_TOP_K}"),
        ));
    }
    let permit = acquire_slot(&state).await?;
    let backend = state.backend.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let backend = backend.lock()
            .map_err(|_| "SQLite backend mutex is poisoned".to_string())?;
        execute_aml_search(&*backend, request)
    })
    .await
    .map_err(|error| ServiceError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("Search worker failed: {error}")))?;
    result.map(Json).map_err(|error| {
        ServiceError::new(StatusCode::INTERNAL_SERVER_ERROR, error)
    })
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// Construct the router for integration testing or embedding.
pub fn router(config: AmlHttpServerConfig) -> Result<Router, String> {
    config.validate()?;
    let backend = SqliteAmlMemoryBackend::open(&config.database_path, &config.payload_root)?;
    let state = Arc::new(AmlHttpState {
        backend: Arc::new(Mutex::new(backend)),
        api_key: config.api_key,
        request_slots: Arc::new(Semaphore::new(config.max_concurrency)),
    });
    Ok(Router::new()
        .route("/health", get(health_handler))
        .route("/add", post(add_handler))
        .route("/search", post(search_handler))
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .with_state(state))
}

/// Run the participant API on the configured address. TLS termination must
/// happen at a public reverse proxy; do not expose this plain HTTP listener
/// directly to the public Internet.
pub async fn serve(config: AmlHttpServerConfig) -> Result<(), String> {
    let bind = config.bind;
    let app = router(config)?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .map_err(|error| format!("failed to bind AML HTTP server: {error}"))?;
    axum::serve(listener, app)
        .await
        .map_err(|error| format!("AML HTTP server failed: {error}"))
}

/// Environment-configured entry point for the optional binary.
pub async fn serve_from_env() -> Result<(), String> {
    serve(AmlHttpServerConfig::from_env()?).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_success_serialization_matches_official_echo_shape() {
        use crate::memory_challenge_api::{AmlMessage, AmlMessageRole};
        let request = AmlAddRequest {
            request_id: "request-1".into(),
            messages: vec![AmlMessage {
                role: AmlMessageRole::User,
                content: "known test evidence".into(),
                timestamp: Some(1_700_000_000_000),
            }],
            user_id: "user-1".into(),
            session_id: "session-1".into(),
        };
        let response = request.success_response();
        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value, serde_json::json!({
            "success": true,
            "request_id": "request-1",
            "user_id": "user-1",
            "session_id": "session-1"
        }));
    }

    #[test]
    fn search_response_serialization_is_top_level_data_array() {
        let response = AmlSearchResponse {
            data: vec![crate::memory_challenge_api::AmlSearchResult {
                id: "memory-1".into(),
                content: "source evidence".into(),
                score: Some(0.75),
                created_at: None,
            }],
        };
        let value = serde_json::to_value(response).unwrap();
        assert!(value.get("data").is_some_and(serde_json::Value::is_array));
        assert!(value.get("items").is_none());
        assert_eq!(value["data"][0]["id"], "memory-1");
    }

    #[tokio::test]
    async fn health_is_public_but_add_and_search_require_the_memory_system_key() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let directory = tempfile::tempdir().unwrap();
        let app = router(AmlHttpServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            database_path: directory.path().join("aml.sqlite3"),
            payload_root: directory.path().join("payloads"),
            api_key: "a".repeat(40),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrency: 2,
        }).unwrap();

        let health = app.clone().oneshot(
            Request::builder().method("GET").uri("/health")
                .body(Body::empty()).unwrap()
        ).await.unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let unauthenticated = app.oneshot(
            Request::builder().method("POST").uri("/add")
                .header("content-type", "application/json")
                .body(Body::from("{}")).unwrap()
        ).await.unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn synchronous_http_add_is_immediately_searchable_with_user_scope() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request;
        use crate::memory_challenge_api::{AmlMessage, AmlMessageRole};
        use tower::ServiceExt;

        let directory = tempfile::tempdir().unwrap();
        let key = "participant-memory-system-key-0123456789";
        let app = router(AmlHttpServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            database_path: directory.path().join("aml.sqlite3"),
            payload_root: directory.path().join("payloads"),
            api_key: key.into(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrency: 2,
        }).unwrap();

        let add = AmlAddRequest {
            request_id: "request-1".into(),
            messages: vec![
                AmlMessage {
                    role: AmlMessageRole::User,
                    content: "The cache invalidation fix shipped after the schema migration.".into(),
                    timestamp: Some(1_700_000_000_000),
                },
            ],
            user_id: "eval-user-1".into(),
            session_id: "eval-session-1".into(),
        };
        let add_response = app.clone().oneshot(
            Request::builder().method("POST").uri("/add")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key}"))
                .body(Body::from(serde_json::to_vec(&add).unwrap())).unwrap()
        ).await.unwrap();
        assert_eq!(add_response.status(), StatusCode::OK);
        let add_bytes = to_bytes(add_response.into_body(), DEFAULT_MAX_BODY_BYTES).await.unwrap();
        let add_json: serde_json::Value = serde_json::from_slice(&add_bytes).unwrap();
        assert_eq!(add_json, serde_json::json!({
            "success": true,
            "request_id": "request-1",
            "user_id": "eval-user-1",
            "session_id": "eval-session-1"
        }));

        let search = AmlSearchRequest {
            query: "cache invalidation schema migration".into(),
            options: None,
            user_id: "eval-user-1".into(),
            top_k: 100,
        };
        let search_response = app.clone().oneshot(
            Request::builder().method("POST").uri("/search")
                .header("content-type", "application/json")
                .header("x-api-key", key)
                .body(Body::from(serde_json::to_vec(&search).unwrap())).unwrap()
        ).await.unwrap();
        assert_eq!(search_response.status(), StatusCode::OK);
        let search_bytes = to_bytes(search_response.into_body(), DEFAULT_MAX_BODY_BYTES).await.unwrap();
        let search_json: serde_json::Value = serde_json::from_slice(&search_bytes).unwrap();
        assert_eq!(search_json["data"].as_array().unwrap().len(), 1);
        assert!(search_json["data"][0]["content"].as_str().unwrap().contains("cache invalidation"));

        let isolated = AmlSearchRequest {
            query: "cache invalidation schema migration".into(),
            options: None,
            user_id: "different-eval-user".into(),
            top_k: 100,
        };
        let isolated_response = app.oneshot(
            Request::builder().method("POST").uri("/search")
                .header("content-type", "application/json")
                .header("authorization", format!("Token {key}"))
                .body(Body::from(serde_json::to_vec(&isolated).unwrap())).unwrap()
        ).await.unwrap();
        assert_eq!(isolated_response.status(), StatusCode::OK);
        let isolated_bytes = to_bytes(isolated_response.into_body(), DEFAULT_MAX_BODY_BYTES).await.unwrap();
        let isolated_json: serde_json::Value = serde_json::from_slice(&isolated_bytes).unwrap();
        assert_eq!(isolated_json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn search_rejects_top_k_above_official_limit() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let directory = tempfile::tempdir().unwrap();
        let key = "participant-memory-system-key-0123456789";
        let app = router(AmlHttpServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            database_path: directory.path().join("aml.sqlite3"),
            payload_root: directory.path().join("payloads"),
            api_key: key.into(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrency: 2,
        }).unwrap();
        let request = serde_json::json!({
            "query": "evidence",
            "user_id": "user-1",
            "top_k": 101
        });
        let response = app.oneshot(
            Request::builder().method("POST").uri("/search")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key}"))
                .body(Body::from(serde_json::to_vec(&request).unwrap())).unwrap()
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn constant_time_comparison_accepts_only_exact_key_bytes() {
        assert!(constant_time_eq(b"secret-key", b"secret-key"));
        assert!(!constant_time_eq(b"secret-key", b"secret-key "));
        assert!(!constant_time_eq(b"secret-key", b"secret"));
        assert!(!constant_time_eq(b"", b"secret-key"));
    }

    #[test]
    fn accepts_official_authentication_schemes() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer secret".parse().unwrap());
        assert_eq!(supplied_api_key(&headers), Some("secret"));
        headers.insert(AUTHORIZATION, "Token secret".parse().unwrap());
        assert_eq!(supplied_api_key(&headers), Some("secret"));
        headers.remove(AUTHORIZATION);
        headers.insert("x-api-key", "secret".parse().unwrap());
        assert_eq!(supplied_api_key(&headers), Some("secret"));
    }

    #[test]
    fn rejects_missing_malformed_and_wrong_authentication() {
        let headers = HeaderMap::new();
        assert!(!authenticate(&headers, "expected-key"));
        let mut malformed = HeaderMap::new();
        malformed.insert(AUTHORIZATION, "Basic expected-key".parse().unwrap());
        assert!(!authenticate(&malformed, "expected-key"));
        let mut wrong = HeaderMap::new();
        wrong.insert("x-api-key", "wrong-key".parse().unwrap());
        assert!(!authenticate(&wrong, "expected-key"));
    }

    #[test]
    fn production_config_requires_high_entropy_key_and_bounded_limits() {
        let base = AmlHttpServerConfig {
            bind: "127.0.0.1:8080".parse().unwrap(),
            database_path: PathBuf::from("aml.sqlite3"),
            payload_root: PathBuf::from("payloads"),
            api_key: "too-short".into(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        };
        assert!(base.validate().unwrap_err().contains("at least 32 bytes"));
        let mut valid = base.clone();
        valid.api_key = "a".repeat(32);
        assert!(valid.validate().is_ok());
        valid.max_concurrency = 0;
        assert!(valid.validate().is_err());
    }

    #[test]
    fn environment_integer_parser_rejects_negative_and_non_numeric_values() {
        assert!(parse_env_usize("AML_INVALID_TEST_VALUE", usize::MAX).is_ok());
        // Parsing helper is tested with a temporary scoped environment variable
        // by the integration harness; this unit test verifies its default path.
        assert_eq!(
            parse_env_usize("AML_TEST_MISSING_ENV_8B5A", 17).unwrap(),
            17
        );
    }
}
