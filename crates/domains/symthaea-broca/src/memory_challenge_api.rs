// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Agent Memory Challenge (AML) Add/Search API contract types for Text/Coding.
//!
//! This module intentionally defines the narrow participant boundary only.
//! It does not start an HTTP server, persist requests, authenticate callers,
//! or claim conformance until a deployed service passes the official Smoke test.
//!
//! Critical contract invariants:
//! - Add preserves message order and echoes request_id/user_id/session_id exactly.
//! - Add success is returned only after durable write and immediate Search visibility.
//! - Retries reuse the same request_id and identical payload.
//! - Search is scoped only by user_id (never by session_id) and returns evidence,
//!   not generated answers or evaluator labels.
//! - Search response order is the retrieval rank order; entries must not exceed top_k.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const AML_TEXT_CODING_API_VERSION: &str = "cycle2-text-coding-v1";

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must be non-empty"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AmlMessageRole {
    User,
    Assistant,
}

/// One ordered source message. AML Text/Coding requires string content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlMessage {
    pub role: AmlMessageRole,
    pub content: String,
    /// Optional Unix timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl AmlMessage {
    fn validate(&self, index: usize) -> Result<(), String> {
        require_non_empty(&self.content, &format!("messages[{index}].content"))
    }
}

/// AML Text/Coding Add request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlAddRequest {
    pub request_id: String,
    /// Preserve source order exactly; do not sort or deduplicate messages.
    pub messages: Vec<AmlMessage>,
    /// The only cross-request memory retrieval isolation key.
    pub user_id: String,
    /// Groups source provenance; this is not a Search filter.
    pub session_id: String,
}

impl AmlAddRequest {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.request_id, "request_id")?;
        require_non_empty(&self.user_id, "user_id")?;
        require_non_empty(&self.session_id, "session_id")?;
        if self.messages.is_empty() {
            return Err("messages must contain at least one source message".into());
        }
        for (index, message) in self.messages.iter().enumerate() {
            message.validate(index)?;
        }
        Ok(())
    }

    /// Stable content digest for idempotency receipts.
    ///
    /// This hashes the canonical serde representation of this typed request;
    /// message order and all supplied timestamp values are included.
    pub fn payload_sha256(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("failed to serialize Add request: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Construct the exact success envelope. Call only after the backing store
    /// has durably committed and a read-after-write Search can see the messages.
    pub fn success_response(&self) -> AmlAddResponse {
        AmlAddResponse {
            success: true,
            request_id: self.request_id.clone(),
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
        }
    }
}

/// Receipt returned by a backend only after its durable commit and immediate
/// read-after-write Search visibility barrier have both completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlAddCommitReceipt {
    pub request_id: String,
    pub payload_sha256: String,
    pub durable: bool,
    pub searchable: bool,
}

/// Backend boundary. Implementations must enforce durable idempotency and
/// isolate Search exclusively by request.user_id. This trait intentionally
/// leaves storage/index implementation choices open.
pub trait AmlMemoryBackend {
    fn commit_add_and_confirm_searchable(
        &mut self,
        request: &AmlAddRequest,
        payload_sha256: &str,
    ) -> Result<AmlAddCommitReceipt, String>;

    fn search_ranked(
        &self,
        request: &AmlSearchRequest,
    ) -> Result<AmlSearchResponse, String>;
}

/// Execute the participant-side Add contract. An HTTP handler may emit 200
/// only after this returns successfully.
pub fn execute_aml_add<B: AmlMemoryBackend>(
    backend: &mut B,
    request: AmlAddRequest,
) -> Result<AmlAddResponse, String> {
    request.validate()?;
    let payload_sha256 = request.payload_sha256()?;
    let receipt = backend.commit_add_and_confirm_searchable(&request, &payload_sha256)?;
    if receipt.request_id != request.request_id {
        return Err("backend commit receipt request_id mismatch".into());
    }
    if receipt.payload_sha256 != payload_sha256 {
        return Err("backend commit receipt payload digest mismatch".into());
    }
    if !receipt.durable {
        return Err("backend has not durably persisted the Add request".into());
    }
    if !receipt.searchable {
        return Err("backend has not made the Add request immediately searchable".into());
    }
    Ok(request.success_response())
}

/// Execute the participant-side Search contract. The backend receives the
/// request with its sole user_id isolation scope; the returned evidence is
/// validated but never rewritten, re-ranked, or converted into an answer.
pub fn execute_aml_search<B: AmlMemoryBackend>(
    backend: &B,
    request: AmlSearchRequest,
) -> Result<AmlSearchResponse, String> {
    request.validate()?;
    let response = backend.search_ranked(&request)?;
    response.validate_for(&request)?;
    Ok(response)
}

/// Successful AML Add response. HTTP status must be 200 for the synchronous
/// contract; this value alone cannot prove that persistence/searchability occurred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlAddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
}

impl AmlAddResponse {
    pub fn validate_echo(&self, request: &AmlAddRequest) -> Result<(), String> {
        if !self.success {
            return Err("successful Add response must set success=true".into());
        }
        if self.request_id != request.request_id {
            return Err("Add response request_id does not exactly match request".into());
        }
        if self.user_id != request.user_id {
            return Err("Add response user_id does not exactly match request".into());
        }
        if self.session_id != request.session_id {
            return Err("Add response session_id does not exactly match request".into());
        }
        Ok(())
    }
}

/// Persistent implementations should store one of these with the Add payload.
/// A retry with an identical digest is a replay; reusing the ID for different
/// content is a conflict. This helper does not itself provide durable storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlAddIdempotencyReceipt {
    pub request_id: String,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmlAddRetryDecision {
    NewRequest,
    ReplayIdenticalRequest,
    ConflictRequestIdReusedWithDifferentPayload,
}

pub fn classify_add_retry(
    existing: Option<&AmlAddIdempotencyReceipt>,
    request: &AmlAddRequest,
) -> Result<AmlAddRetryDecision, String> {
    let payload_sha256 = request.payload_sha256()?;
    let Some(existing) = existing else {
        return Ok(AmlAddRetryDecision::NewRequest);
    };
    if existing.request_id != request.request_id {
        return Err("idempotency receipt request_id does not match incoming request".into());
    }
    if existing.payload_sha256 == payload_sha256 {
        Ok(AmlAddRetryDecision::ReplayIdenticalRequest)
    } else {
        Ok(AmlAddRetryDecision::ConflictRequestIdReusedWithDifferentPayload)
    }
}

/// AML Text/Coding Search request.
///
/// The absence of session_id is deliberate: user_id is the only retrieval
/// isolation parameter. Adding a session filter changes the evaluation contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmlSearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
    pub user_id: String,
    pub top_k: usize,
}

impl AmlSearchRequest {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.query, "query")?;
        require_non_empty(&self.user_id, "user_id")?;
        if self.top_k == 0 {
            return Err("top_k must be greater than zero".into());
        }
        if let Some(options) = &self.options {
            if options.is_empty() {
                return Err("options must be omitted for open questions or contain choices".into());
            }
            for (index, option) in options.iter().enumerate() {
                require_non_empty(option, &format!("options[{index}]"))?;
            }
        }
        Ok(())
    }
}

/// One ranked memory evidence item returned to the platform's Answer stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmlSearchResult {
    pub id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

impl AmlSearchResult {
    fn validate(&self, index: usize) -> Result<(), String> {
        require_non_empty(&self.id, &format!("data[{index}].id"))?;
        require_non_empty(&self.content, &format!("data[{index}].content"))?;
        if self.score.is_some_and(|score| !score.is_finite()) {
            return Err(format!("data[{index}].score must be finite when present"));
        }
        if let Some(created_at) = &self.created_at {
            require_non_empty(created_at, &format!("data[{index}].created_at"))?;
        }
        Ok(())
    }
}

/// AML Search response. Vector order is semantically significant and must
/// already be relevance-ranked by the participant implementation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AmlSearchResponse {
    pub data: Vec<AmlSearchResult>,
}

impl AmlSearchResponse {
    pub fn validate_for(&self, request: &AmlSearchRequest) -> Result<(), String> {
        request.validate()?;
        if self.data.len() > request.top_k {
            return Err(format!(
                "Search returned {} items, exceeding top_k={}",
                self.data.len(),
                request.top_k
            ));
        }
        let mut seen_ids = BTreeSet::new();
        for (index, result) in self.data.iter().enumerate() {
            result.validate(index)?;
            if !seen_ids.insert(result.id.as_str()) {
                return Err(format!("Search returned duplicate memory ID {:?}", result.id));
            }
        }
        // When every candidate supplies a scalar score, its order must agree
        // with the stated higher-is-more-relevant score convention. If scores
        // are absent or partial, preserve the backend's opaque ranked order.
        if self.data.iter().all(|result| result.score.is_some()) {
            for pair in self.data.windows(2) {
                if pair[0].score.expect("all scores present")
                    < pair[1].score.expect("all scores present")
                {
                    return Err("Search scores are not ordered from higher to lower relevance".into());
                }
            }
        }
        Ok(())
    }

    /// Empty results must serialize as {"data":[]}, not null or a missing field.
    pub fn empty() -> Self {
        Self { data: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: AmlMessageRole, content: &str, timestamp: Option<i64>) -> AmlMessage {
        AmlMessage {
            role,
            content: content.into(),
            timestamp,
        }
    }

    fn add_request() -> AmlAddRequest {
        AmlAddRequest {
            request_id: "eval:run-1:repo:session-0:chunk-0".into(),
            messages: vec![
                message(AmlMessageRole::User, "Investigated failing test.", Some(1_704_067_200_000)),
                message(AmlMessageRole::Assistant, "Root cause was a stale cache key.", Some(1_704_067_201_000)),
            ],
            user_id: "eval:run-1:repo-42".into(),
            session_id: "eval:run-1:repo-42:session-0".into(),
        }
    }

    fn search_request(top_k: usize) -> AmlSearchRequest {
        AmlSearchRequest {
            query: "Why did the cache test fail?".into(),
            options: None,
            user_id: "eval:run-1:repo-42".into(),
            top_k,
        }
    }

    fn result(id: &str, content: &str, score: f64) -> AmlSearchResult {
        AmlSearchResult {
            id: id.into(),
            content: content.into(),
            score: Some(score),
            created_at: Some("2026-10-30T09:00:00Z".into()),
        }
    }

    #[test]
    fn add_request_preserves_order_and_round_trips() {
        let request = add_request();
        request.validate().unwrap();
        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: AmlAddRequest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(decoded.messages[0].content, "Investigated failing test.");
        assert_eq!(decoded.messages[1].content, "Root cause was a stale cache key.");
    }

    #[test]
    fn add_rejects_empty_identifiers() {
        let mut request = add_request();
        request.request_id.clear();
        assert!(request.validate().unwrap_err().contains("request_id"));
        request = add_request();
        request.user_id = "  ".into();
        assert!(request.validate().unwrap_err().contains("user_id"));
        request = add_request();
        request.session_id.clear();
        assert!(request.validate().unwrap_err().contains("session_id"));
    }

    #[test]
    fn add_rejects_empty_message_array() {
        let mut request = add_request();
        request.messages.clear();
        assert!(request.validate().unwrap_err().contains("at least one"));
    }

    #[test]
    fn add_rejects_empty_message_content() {
        let mut request = add_request();
        request.messages[1].content = " ".into();
        assert!(request.validate().unwrap_err().contains("messages[1].content"));
    }

    #[test]
    fn add_message_timestamp_is_optional() {
        let mut request = add_request();
        request.messages[0].timestamp = None;
        request.validate().unwrap();
    }

    #[test]
    fn add_request_payload_digest_is_deterministic() {
        let request = add_request();
        assert_eq!(request.payload_sha256().unwrap(), request.payload_sha256().unwrap());
    }

    #[test]
    fn add_message_order_changes_payload_digest() {
        let mut request = add_request();
        let original = request.payload_sha256().unwrap();
        request.messages.reverse();
        assert_ne!(request.payload_sha256().unwrap(), original);
    }

    #[test]
    fn add_retry_with_same_id_and_same_payload_is_idempotent_replay() {
        let request = add_request();
        let receipt = AmlAddIdempotencyReceipt {
            request_id: request.request_id.clone(),
            payload_sha256: request.payload_sha256().unwrap(),
        };
        assert_eq!(
            classify_add_retry(Some(&receipt), &request).unwrap(),
            AmlAddRetryDecision::ReplayIdenticalRequest
        );
    }

    #[test]
    fn add_retry_with_same_id_and_different_payload_is_conflict() {
        let original = add_request();
        let receipt = AmlAddIdempotencyReceipt {
            request_id: original.request_id.clone(),
            payload_sha256: original.payload_sha256().unwrap(),
        };
        let mut changed = original;
        changed.messages[0].content.push_str(" altered");
        assert_eq!(
            classify_add_retry(Some(&receipt), &changed).unwrap(),
            AmlAddRetryDecision::ConflictRequestIdReusedWithDifferentPayload
        );
    }

    #[test]
    fn add_retry_with_unrelated_receipt_is_rejected() {
        let request = add_request();
        let receipt = AmlAddIdempotencyReceipt {
            request_id: "other-id".into(),
            payload_sha256: request.payload_sha256().unwrap(),
        };
        assert!(classify_add_retry(Some(&receipt), &request).is_err());
    }

    #[test]
    fn add_success_response_echoes_all_ids_exactly() {
        let request = add_request();
        let response = request.success_response();
        response.validate_echo(&request).unwrap();
        assert_eq!(response.request_id.as_bytes(), request.request_id.as_bytes());
        assert_eq!(response.user_id.as_bytes(), request.user_id.as_bytes());
        assert_eq!(response.session_id.as_bytes(), request.session_id.as_bytes());
    }

    #[test]
    fn add_echo_validation_rejects_mismatched_session() {
        let request = add_request();
        let mut response = request.success_response();
        response.session_id.push_str("-other");
        assert!(response.validate_echo(&request).unwrap_err().contains("session_id"));
    }

    #[test]
    fn add_response_cannot_claim_failure_as_success() {
        let request = add_request();
        let mut response = request.success_response();
        response.success = false;
        assert!(response.validate_echo(&request).is_err());
    }

    #[test]
    fn search_request_is_user_scoped_and_has_no_session_filter() {
        let request = search_request(100);
        request.validate().unwrap();
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["user_id"], "eval:run-1:repo-42");
        assert!(json.get("session_id").is_none());
    }

    #[test]
    fn search_rejects_empty_query_and_user_scope() {
        let mut request = search_request(100);
        request.query.clear();
        assert!(request.validate().unwrap_err().contains("query"));
        request = search_request(100);
        request.user_id.clear();
        assert!(request.validate().unwrap_err().contains("user_id"));
    }

    #[test]
    fn search_rejects_zero_top_k() {
        assert!(search_request(0).validate().unwrap_err().contains("top_k"));
    }

    #[test]
    fn search_options_are_optional_but_must_be_nonempty_when_supplied() {
        search_request(100).validate().unwrap();
        let mut request = search_request(100);
        request.options = Some(vec!["A".into(), "B".into()]);
        request.validate().unwrap();
        request.options = Some(Vec::new());
        assert!(request.validate().unwrap_err().contains("options"));
    }

    #[test]
    fn empty_search_result_uses_required_data_array() {
        let request = search_request(100);
        let response = AmlSearchResponse::empty();
        response.validate_for(&request).unwrap();
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["data"], serde_json::json!([]));
    }

    #[test]
    fn search_response_preserves_retrieval_order() {
        let request = search_request(2);
        let response = AmlSearchResponse {
            data: vec![result("mem-2", "Higher ranked memory", 0.92), result("mem-1", "Second memory", 0.81)],
        };
        response.validate_for(&request).unwrap();
        assert_eq!(response.data[0].id, "mem-2");
        assert_eq!(response.data[1].id, "mem-1");
    }

    #[test]
    fn search_response_rejects_more_than_top_k_without_truncating() {
        let request = search_request(1);
        let response = AmlSearchResponse {
            data: vec![result("mem-1", "first", 0.9), result("mem-2", "second", 0.8)],
        };
        assert!(response.validate_for(&request).unwrap_err().contains("exceeding top_k"));
        assert_eq!(response.data.len(), 2);
    }

    #[test]
    fn search_response_rejects_empty_id_content_and_duplicate_ids() {
        let request = search_request(10);
        let blank_id = AmlSearchResponse { data: vec![result("", "memory", 0.8)] };
        assert!(blank_id.validate_for(&request).unwrap_err().contains("id"));
        let blank_content = AmlSearchResponse { data: vec![result("mem-1", " ", 0.8)] };
        assert!(blank_content.validate_for(&request).unwrap_err().contains("content"));
        let duplicate = AmlSearchResponse {
            data: vec![result("mem-1", "one", 0.9), result("mem-1", "duplicate", 0.8)],
        };
        assert!(duplicate.validate_for(&request).unwrap_err().contains("duplicate memory ID"));
    }

    #[test]
    fn search_response_ignores_undeclared_fields_as_platform_does() {
        let json = serde_json::json!({
            "data": [{
                "id": "mem-1",
                "content": "retrieved evidence",
                "score": 0.9,
                "platform_ignored_field": "ignored"
            }],
            "platform_response_metadata": "ignored"
        });
        let response: AmlSearchResponse = serde_json::from_value(json).unwrap();
        response.validate_for(&search_request(10)).unwrap();
        assert_eq!(response.data[0].id, "mem-1");
    }

    #[test]
    fn search_response_rejects_score_order_inversion_when_all_scores_are_present() {
        let request = search_request(10);
        let response = AmlSearchResponse {
            data: vec![result("mem-1", "less relevant", 0.7), result("mem-2", "more relevant", 0.9)],
        };
        assert!(response.validate_for(&request).unwrap_err().contains("ordered from higher to lower"));
    }

    #[test]
    fn backend_can_preserve_rank_order_when_optional_scores_are_missing() {
        let request = search_request(10);
        let response = AmlSearchResponse {
            data: vec![
                AmlSearchResult { id: "mem-1".into(), content: "rank one".into(), score: Some(0.9), created_at: None },
                AmlSearchResult { id: "mem-2".into(), content: "rank two".into(), score: None, created_at: None },
                AmlSearchResult { id: "mem-3".into(), content: "rank three".into(), score: Some(0.1), created_at: None },
            ],
        };
        response.validate_for(&request).unwrap();
        assert_eq!(response.data[0].id, "mem-1");
    }

    #[test]
    fn search_response_rejects_non_finite_scores() {
        let request = search_request(10);
        let response = AmlSearchResponse {
            data: vec![result("mem-1", "memory", f64::NAN)],
        };
        assert!(response.validate_for(&request).unwrap_err().contains("finite"));
    }

    #[test]
    fn search_response_rejects_empty_created_at_when_supplied() {
        let request = search_request(10);
        let mut item = result("mem-1", "memory", 0.8);
        item.created_at = Some(String::new());
        let response = AmlSearchResponse { data: vec![item] };
        assert!(response.validate_for(&request).unwrap_err().contains("created_at"));
    }
}
