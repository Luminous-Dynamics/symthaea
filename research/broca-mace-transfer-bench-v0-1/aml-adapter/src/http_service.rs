// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, authenticated HTTP surface for the AML Add/Search contract.
//!
//! Routes expose only Add, Search and a public health check. They preserve the
//! protocol types in the parent module and use the existing durable coordinator;
//! there is no second implementation of persistence, idempotency, visibility
//! verification or tenant isolation.
//!
//! The redb backend is single-writer. One semaphore permit is held across each
//! blocking store call, which runs on Tokio's blocking pool so database I/O does
//! not block an async executor worker. Public deployments must terminate TLS
//! before this router or wrap it in a TLS listener.

use std::sync::{Arc, Mutex};

use axum::{
    body::to_bytes,
    extract::{Request, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;

use super::{
    AddRequest, AddResponse, ContentPart, DurableAmlServiceErrorV1,
    DurableAmlServiceV1, DurableAmlStoreV1, FrozenAdapterWordCounter,
    Message, MessageContent, Role, SearchRequest, SearchResponse, Track,
};

/// Base64 expands the official 30 MiB aggregate Add image allowance to 40 MiB.
/// The 48 MiB cap leaves room for JSON framing without accepting unbounded
/// bodies. Deployments may lower this limit; they must not silently truncate.
pub const DEFAULT_MAX_JSON_BODY_BYTES: usize = 48 * 1024 * 1024;

/// These schemes are mutually exclusive. PublicSmoke is explicit and must
/// never be selected for a production deployment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthSchemeV1 {
    Token,
    Bearer,
    XApiKey,
    PublicSmoke,
}

#[derive(Clone)]
pub struct ApiAuthV1 {
    scheme: AuthSchemeV1,
    expected_secret_digest: Option<[u8; 32]>,
}

impl ApiAuthV1 {
    pub fn from_secret(
        scheme: AuthSchemeV1,
        secret: &str,
    ) -> Result<Self, HttpConfigErrorV1> {
        if scheme == AuthSchemeV1::PublicSmoke || secret.is_empty() {
            return Err(HttpConfigErrorV1::InvalidAuthentication);
        }
        let digest = Sha256::digest(secret.as_bytes());
        let mut expected = [0_u8; 32];
        expected.copy_from_slice(&digest);
        Ok(Self {
            scheme,
            expected_secret_digest: Some(expected),
        })
    }

    /// Explicitly enables unauthenticated requests for public Smoke only.
    pub fn public_smoke() -> Self {
        Self {
            scheme: AuthSchemeV1::PublicSmoke,
            expected_secret_digest: None,
        }
    }

    fn authorize(&self, headers: &HeaderMap) -> bool {
        if self.scheme == AuthSchemeV1::PublicSmoke {
            return true;
        }
        let candidate = match self.scheme {
            AuthSchemeV1::Token => headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| strip_auth_scheme(value, "Token")),
            AuthSchemeV1::Bearer => headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| strip_auth_scheme(value, "Bearer")),
            AuthSchemeV1::XApiKey => headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok()),
            AuthSchemeV1::PublicSmoke => return true,
        };
        let Some(candidate) = candidate else {
            return false;
        };
        let Some(expected) = self.expected_secret_digest.as_ref() else {
            return false;
        };
        let actual = Sha256::digest(candidate.as_bytes());
        bool::from(expected.as_slice().ct_eq(actual.as_slice()))
    }
}

pub struct AmlHttpConfigV1 {
    track: Track,
    auth: ApiAuthV1,
    max_json_body_bytes: usize,
}

impl AmlHttpConfigV1 {
    pub fn new(track: Track, auth: ApiAuthV1) -> Result<Self, HttpConfigErrorV1> {
        Self::with_body_limit(track, auth, DEFAULT_MAX_JSON_BODY_BYTES)
    }

    pub fn with_body_limit(
        track: Track,
        auth: ApiAuthV1,
        max_json_body_bytes: usize,
    ) -> Result<Self, HttpConfigErrorV1> {
        if max_json_body_bytes < 1024 {
            return Err(HttpConfigErrorV1::InvalidBodyLimit);
        }
        Ok(Self {
            track,
            auth,
            max_json_body_bytes,
        })
    }

    pub fn track(&self) -> Track {
        self.track
    }

    pub fn max_concurrent_requests(&self) -> usize {
        // The current redb store is a single-writer backend and the coordinator
        // shares one service mutex. Do not advertise more than one request.
        1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpConfigErrorV1 {
    InvalidAuthentication,
    InvalidBodyLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HttpInputErrorV1 {
    InvalidObject,
    UnknownField,
    MissingField,
    WrongType,
    InvalidRole,
    InvalidTimestamp,
    InvalidContentPart,
}

struct HttpStateV1<S, V, C> {
    service: Arc<Mutex<DurableAmlServiceV1<S, V, C>>>,
    auth: ApiAuthV1,
    track: Track,
    request_permit: Arc<Semaphore>,
    max_json_body_bytes: usize,
}

impl<S, V, C> Clone for HttpStateV1<S, V, C> {
    fn clone(&self) -> Self {
        Self {
            service: Arc::clone(&self.service),
            auth: self.auth.clone(),
            track: self.track,
            request_permit: Arc::clone(&self.request_permit),
            max_json_body_bytes: self.max_json_body_bytes,
        }
    }
}

/// Build the service router. HTTPS must be provided by the listener or public
/// reverse proxy. The router enforces one in-flight request to match the
/// current redb service's honest single-writer capacity.
pub fn router<S, V, C>(
    service: DurableAmlServiceV1<S, V, C>,
    config: AmlHttpConfigV1,
) -> Router
where
    S: DurableAmlStoreV1 + Send + 'static,
    V: super::AddVisibilityVerifierV1 + Send + 'static,
    C: FrozenAdapterWordCounter + Send + 'static,
{
    let body_limit = config.max_json_body_bytes;
    let state = HttpStateV1 {
        service: Arc::new(Mutex::new(service)),
        auth: config.auth,
        track: config.track,
        request_permit: Arc::new(Semaphore::new(1)),
        max_json_body_bytes: body_limit,
    };

    Router::new()
        .route("/health", get(health))
        .route("/add", post(add_handler::<S, V, C>))
        .route("/search", post(search_handler::<S, V, C>))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn add_handler<S, V, C>(
    State(state): State<HttpStateV1<S, V, C>>,
    request: Request,
) -> Response
where
    S: DurableAmlStoreV1 + Send + 'static,
    V: super::AddVisibilityVerifierV1 + Send + 'static,
    C: FrozenAdapterWordCounter + Send + 'static,
{
    if !state.auth.authorize(request.headers()) {
        return api_error(StatusCode::UNAUTHORIZED, "authentication_required", None);
    }
    if !is_json_content_type(request.headers()) {
        return api_error(StatusCode::UNSUPPORTED_MEDIA_TYPE, "application_json_required", None);
    }
    let permit = match Arc::clone(&state.request_permit).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return api_error(StatusCode::TOO_MANY_REQUESTS, "service_at_capacity", Some(1)),
    };
    let bytes = match to_bytes(request.into_body(), state.max_json_body_bytes).await {
        Ok(bytes) => bytes,
        Err(_) => return api_error(StatusCode::PAYLOAD_TOO_LARGE, "request_body_rejected", None),
    };
    let value = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value,
        Err(_) => return api_error(StatusCode::BAD_REQUEST, "invalid_json", None),
    };
    let request = match parse_add_request(value) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::UNPROCESSABLE_ENTITY, input_error_code(error), None),
    };
    let service = Arc::clone(&state.service);
    let track = state.track;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut service = service.lock().map_err(|_| ())?;
        Ok::<_, ()>(service.add(track, &request))
    })
    .await;

    match result {
        Ok(Ok(Ok(response))) => Json(add_response_json(response)).into_response(),
        Ok(Ok(Err(error))) => service_error_response(error),
        Ok(Err(())) | Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", Some(1)),
    }
}

async fn search_handler<S, V, C>(
    State(state): State<HttpStateV1<S, V, C>>,
    request: Request,
) -> Response
where
    S: DurableAmlStoreV1 + Send + 'static,
    V: super::AddVisibilityVerifierV1 + Send + 'static,
    C: FrozenAdapterWordCounter + Send + 'static,
{
    if !state.auth.authorize(request.headers()) {
        return api_error(StatusCode::UNAUTHORIZED, "authentication_required", None);
    }
    if !is_json_content_type(request.headers()) {
        return api_error(StatusCode::UNSUPPORTED_MEDIA_TYPE, "application_json_required", None);
    }
    let permit = match Arc::clone(&state.request_permit).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return api_error(StatusCode::TOO_MANY_REQUESTS, "service_at_capacity", Some(1)),
    };
    let bytes = match to_bytes(request.into_body(), state.max_json_body_bytes).await {
        Ok(bytes) => bytes,
        Err(_) => return api_error(StatusCode::PAYLOAD_TOO_LARGE, "request_body_rejected", None),
    };
    let value = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value,
        Err(_) => return api_error(StatusCode::BAD_REQUEST, "invalid_json", None),
    };
    let request = match parse_search_request(value) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::UNPROCESSABLE_ENTITY, input_error_code(error), None),
    };
    let service = Arc::clone(&state.service);
    let track = state.track;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut service = service.lock().map_err(|_| ())?;
        Ok::<_, ()>(service.search(track, &request))
    })
    .await;

    match result {
        Ok(Ok(Ok(response))) => Json(search_response_json(response)).into_response(),
        Ok(Ok(Err(error))) => service_error_response(error),
        Ok(Err(())) | Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", Some(1)),
    }
}

/// HTTP authentication scheme names are case-insensitive; credential values are not.
fn strip_auth_scheme<'a>(header_value: &'a str, expected_scheme: &str) -> Option<&'a str> {
    let (scheme, credential) = header_value.split_once(' ')?;
    if scheme.eq_ignore_ascii_case(expected_scheme) && !credential.is_empty() {
        Some(credential)
    } else {
        None
    }
}

fn is_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(';').next().is_some_and(|media| {
                media.trim().eq_ignore_ascii_case("application/json")
            })
        })
}

fn api_error(
    status: StatusCode,
    reason: &'static str,
    retry_after_seconds: Option<u64>,
) -> Response {
    let mut response = (
        status,
        Json(json!({ "detail": { "reason": reason } })),
    )
        .into_response();
    if let Some(seconds) = retry_after_seconds {
        if let Ok(value) = HeaderValue::from_str(&seconds.to_string()) {
            response.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    response
}

fn service_error_response(error: DurableAmlServiceErrorV1) -> Response {
    match error {
        // A payload rebound to an existing request_id is a permanent
        // caller conflict, not an ambiguous/in-flight write. Do not advertise
        // it as retryable with Retry-After.
        DurableAmlServiceErrorV1::Contract(super::AmlReject::RequestIdPayloadConflict) => {
            api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "request_id_payload_conflict",
                None,
            )
        }
        DurableAmlServiceErrorV1::AddStillInFlight => {
            api_error(StatusCode::CONFLICT, "add_still_in_flight", Some(1))
        }
        DurableAmlServiceErrorV1::CommitOutcomeUnknown => {
            api_error(StatusCode::CONFLICT, "commit_outcome_unknown", Some(1))
        }
        DurableAmlServiceErrorV1::SearchScopeViolation => {
            api_error(StatusCode::FORBIDDEN, "search_scope_violation", None)
        }
        DurableAmlServiceErrorV1::SearchVisibilityMismatch => {
            api_error(StatusCode::SERVICE_UNAVAILABLE, "search_visibility_unverified", Some(1))
        }
        DurableAmlServiceErrorV1::Contract(_) => {
            api_error(StatusCode::UNPROCESSABLE_ENTITY, "request_contract_rejected", None)
        }
        DurableAmlServiceErrorV1::Store(_) => {
            api_error(StatusCode::SERVICE_UNAVAILABLE, "storage_unavailable", Some(1))
        }
    }
}

fn input_error_code(error: HttpInputErrorV1) -> &'static str {
    match error {
        HttpInputErrorV1::InvalidObject => "invalid_object",
        HttpInputErrorV1::UnknownField => "unknown_field",
        HttpInputErrorV1::MissingField => "missing_field",
        HttpInputErrorV1::WrongType => "wrong_type",
        HttpInputErrorV1::InvalidRole => "invalid_role",
        HttpInputErrorV1::InvalidTimestamp => "invalid_timestamp",
        HttpInputErrorV1::InvalidContentPart => "invalid_content_part",
    }
}

fn strict_object<'a>(
    value: &'a Value,
    allowed_fields: &[&str],
) -> Result<&'a Map<String, Value>, HttpInputErrorV1> {
    let object = value.as_object().ok_or(HttpInputErrorV1::InvalidObject)?;
    if object.keys().any(|key| !allowed_fields.contains(&key.as_str())) {
        return Err(HttpInputErrorV1::UnknownField);
    }
    Ok(object)
}

fn required_string(
    object: &Map<String, Value>,
    key: &str,
) -> Result<String, HttpInputErrorV1> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            if object.contains_key(key) {
                HttpInputErrorV1::WrongType
            } else {
                HttpInputErrorV1::MissingField
            }
        })
}

fn parse_role(value: &str) -> Result<Role, HttpInputErrorV1> {
    match value {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        _ => Err(HttpInputErrorV1::InvalidRole),
    }
}

fn parse_content(value: &Value) -> Result<MessageContent, HttpInputErrorV1> {
    match value {
        Value::String(text) => Ok(MessageContent::Text(text.clone())),
        Value::Array(parts) => {
            let mut parsed = Vec::with_capacity(parts.len());
            for part in parts {
                let object = part.as_object().ok_or(HttpInputErrorV1::InvalidContentPart)?;
                let kind = object
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or(HttpInputErrorV1::InvalidContentPart)?;
                match kind {
                    "text" => {
                        if object.keys().any(|key| key != "type" && key != "text") {
                            return Err(HttpInputErrorV1::UnknownField);
                        }
                        let text = required_string(object, "text")?;
                        parsed.push(ContentPart::Text(text));
                    }
                    "image_url" => {
                        if object.keys().any(|key| key != "type" && key != "image_url") {
                            return Err(HttpInputErrorV1::UnknownField);
                        }
                        let image = object
                            .get("image_url")
                            .ok_or(HttpInputErrorV1::MissingField)?;
                        let image_object = strict_object(image, &["url"])?;
                        let url = required_string(image_object, "url")?;
                        parsed.push(ContentPart::ImageDataUri(url));
                    }
                    _ => return Err(HttpInputErrorV1::InvalidContentPart),
                }
            }
            Ok(MessageContent::Parts(parsed))
        }
        _ => Err(HttpInputErrorV1::WrongType),
    }
}

fn parse_add_request(value: Value) -> Result<AddRequest, HttpInputErrorV1> {
    let object = strict_object(&value, &["request_id", "messages", "user_id", "session_id"])?;
    let request_id = required_string(object, "request_id")?;
    let user_id = required_string(object, "user_id")?;
    let session_id = required_string(object, "session_id")?;
    let messages_value = object.get("messages").ok_or(HttpInputErrorV1::MissingField)?;
    let messages_array = messages_value.as_array().ok_or(HttpInputErrorV1::WrongType)?;
    if messages_array.is_empty() {
        return Err(HttpInputErrorV1::WrongType);
    }

    let mut messages = Vec::with_capacity(messages_array.len());
    for message in messages_array {
        let message_object = strict_object(message, &["role", "content", "timestamp"])?;
        let role = parse_role(&required_string(message_object, "role")?)?;
        let content_value = message_object.get("content").ok_or(HttpInputErrorV1::MissingField)?;
        let content = parse_content(content_value)?;
        let timestamp_ms = match message_object.get("timestamp") {
            Some(value) => Some(value.as_i64().ok_or(HttpInputErrorV1::InvalidTimestamp)?),
            None => None,
        };
        messages.push(Message {
            role,
            timestamp_ms,
            content,
        });
    }
    Ok(AddRequest {
        request_id,
        messages,
        user_id,
        session_id,
    })
}

fn parse_search_request(value: Value) -> Result<SearchRequest, HttpInputErrorV1> {
    let object = strict_object(&value, &["query", "options", "user_id", "top_k"])?;
    let query_value = object.get("query").ok_or(HttpInputErrorV1::MissingField)?;
    let query = parse_content(query_value)?;
    let user_id = required_string(object, "user_id")?;
    let top_k_value = object.get("top_k").ok_or(HttpInputErrorV1::MissingField)?;
    let top_k_u64 = top_k_value.as_u64().ok_or(HttpInputErrorV1::WrongType)?;
    let top_k = usize::try_from(top_k_u64).map_err(|_| HttpInputErrorV1::WrongType)?;
    let options = match object.get("options") {
        None => None,
        Some(value) => {
            let values = value.as_array().ok_or(HttpInputErrorV1::WrongType)?;
            let mut options = Vec::with_capacity(values.len());
            for option in values {
                let text = option.as_str().ok_or(HttpInputErrorV1::WrongType)?;
                options.push(text.to_owned());
            }
            Some(options)
        }
    };
    Ok(SearchRequest {
        query,
        options,
        user_id,
        top_k,
    })
}

fn add_response_json(response: AddResponse) -> Value {
    json!({
        "success": response.success,
        "request_id": response.request_id,
        "user_id": response.user_id,
        "session_id": response.session_id
    })
}

fn search_response_json(response: SearchResponse) -> Value {
    let data: Vec<Value> = response
        .data
        .into_iter()
        .map(|item| {
            let mut object = Map::new();
            object.insert("id".into(), json!(item.id));
            object.insert("content".into(), super::content_to_json(&item.content));
            if let Some(score) = item.score {
                object.insert("score".into(), json!(score));
            }
            if let Some(created_at) = item.created_at {
                object.insert("created_at".into(), json!(created_at));
            }
            Value::Object(object)
        })
        .collect();
    json!({ "data": data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_and_bearer_auth_are_scheme_specific_and_secret_bound() {
        let token = ApiAuthV1::from_secret(AuthSchemeV1::Token, "key-secret").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Token key-secret"));
        assert!(token.authorize(&headers));
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer key-secret"));
        assert!(!token.authorize(&headers));

        let bearer = ApiAuthV1::from_secret(AuthSchemeV1::Bearer, "key-secret").unwrap();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer key-secret"));
        assert!(bearer.authorize(&headers));

        let x_api = ApiAuthV1::from_secret(AuthSchemeV1::XApiKey, "key-secret").unwrap();
        headers.remove(AUTHORIZATION);
        headers.insert("x-api-key", HeaderValue::from_static("key-secret"));
        assert!(x_api.authorize(&headers));
    }

    #[test]
    fn authentication_scheme_is_case_insensitive_but_credential_is_exact() {
        let token = ApiAuthV1::from_secret(AuthSchemeV1::Token, "Key-Secret").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("token Key-Secret"));
        assert!(token.authorize(&headers));
        headers.insert(AUTHORIZATION, HeaderValue::from_static("TOKEN key-secret"));
        assert!(!token.authorize(&headers));

        let bearer = ApiAuthV1::from_secret(AuthSchemeV1::Bearer, "Key-Secret").unwrap();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("bEaReR Key-Secret"));
        assert!(bearer.authorize(&headers));
    }

    #[test]
    fn permanent_request_id_payload_conflict_is_not_advertised_as_retryable() {
        let response = service_error_response(DurableAmlServiceErrorV1::Contract(
            super::AmlReject::RequestIdPayloadConflict,
        ));
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(response.headers().get(RETRY_AFTER).is_none());

        let in_flight = service_error_response(DurableAmlServiceErrorV1::AddStillInFlight);
        assert_eq!(in_flight.status(), StatusCode::CONFLICT);
        assert_eq!(in_flight.headers().get(RETRY_AFTER).unwrap(), "1");
    }

    #[test]
    fn public_smoke_is_only_enabled_by_explicit_configuration() {
        let public = ApiAuthV1::public_smoke();
        assert!(public.authorize(&HeaderMap::new()));
        assert!(ApiAuthV1::from_secret(AuthSchemeV1::PublicSmoke, "ignored").is_err());
        assert!(ApiAuthV1::from_secret(AuthSchemeV1::Token, "").is_err());
    }

    #[test]
    fn strict_add_parser_preserves_timestamp_and_ordered_multimodal_parts() {
        let value = json!({
            "request_id": "r-1",
            "user_id": "u-1",
            "session_id": "s-1",
            "messages": [{
                "role": "user",
                "timestamp": 1704067200000_i64,
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]
            }]
        });
        let parsed = parse_add_request(value).unwrap();
        assert_eq!(parsed.messages[0].timestamp_ms, Some(1_704_067_200_000));
        let MessageContent::Parts(parts) = &parsed.messages[0].content else {
            panic!("ordered parts should be preserved");
        };
        assert!(matches!(parts[0], ContentPart::Text(_)));
        assert!(matches!(parts[1], ContentPart::ImageDataUri(_)));
    }

    #[test]
    fn strict_parsers_reject_unknown_fields_and_invalid_shapes() {
        let add = json!({
            "request_id": "r-1", "user_id": "u-1", "session_id": "s-1",
            "metadata": {}, "messages": [{"role":"user","content":"x"}]
        });
        assert_eq!(parse_add_request(add), Err(HttpInputErrorV1::UnknownField));

        let search = json!({
            "query": "question", "user_id": "u-1", "top_k": 101, "gold_answer": "leak"
        });
        assert_eq!(parse_search_request(search), Err(HttpInputErrorV1::UnknownField));
    }

    #[test]
    fn api_serialization_uses_exact_add_and_search_envelopes() {
        let add = AddResponse {
            success: true,
            request_id: "r".into(),
            user_id: "u".into(),
            session_id: "s".into(),
        };
        assert_eq!(
            add_response_json(add),
            json!({"success":true,"request_id":"r","user_id":"u","session_id":"s"})
        );
        let response = SearchResponse { data: Vec::new() };
        assert_eq!(search_response_json(response), json!({"data":[]}));
    }

    #[test]
    fn retryable_http_errors_have_bounded_retry_after() {
        let response = api_error(StatusCode::TOO_MANY_REQUESTS, "service_at_capacity", Some(1));
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "1");
    }
}
