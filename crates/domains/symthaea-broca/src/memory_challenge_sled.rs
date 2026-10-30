// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Persistent local research baseline for AML Text/Coding Add/Search.
//!
//! The adapter uses the provenance-aware lexical EvidenceIndex and Sled for
//! durable Add batches. It is intentionally a deterministic BM25 baseline,
//! not the required text-embedding-v4 Open-source Methods configuration and
//! not a production deployment. It does not provide HTTP, TLS, authentication,
//! multi-host replication, or a production-grade backup/restore procedure.
//!
//! Each Add request is one immutable persisted batch. A single-key
//! compare-and-swap binds the request ID to its exact payload digest and
//! records. The tree is flushed before an Add success receipt is emitted.
//! On restart, the in-memory evidence index is reconstructed only from
//! verified persisted batches. Retrieval always scopes EvidenceIndex by
//! request.user_id; session_id is provenance only.
//!
//! The evaluation operator must call the retention purge methods to meet the
//! competition's 30-day evaluation-data deletion rule. No background deletion
//! worker is created by this module.

use crate::memory_challenge_api::{
    AmlAddCommitReceipt, AmlAddIdempotencyReceipt, AmlAddRequest, AmlAddRetryDecision,
    AmlMemoryBackend, AmlSearchRequest, AmlSearchResponse, AmlSearchResult,
    classify_add_retry,
};
use crate::memory_evidence::{
    Bm25Config, EVIDENCE_SCHEMA_VERSION, EvidenceIndex, EvidenceLifecycle, EvidencePayload,
    EvidenceRecord, EvidenceSource,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const ADD_BATCH_TREE: &[u8] = b"aml-add-batches-v1";
const ADD_KEY_PREFIX: &[u8] = b"request:";

/// Immutable persisted batch for one Add idempotency key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedAmlAddBatch {
    request: AmlAddRequest,
    payload_sha256: String,
    stored_at_unix_ms: u64,
    records: Vec<EvidenceRecord>,
}

/// Sled-backed Add persistence plus a rebuildable, process-local lexical index.
pub struct SledAmlMemoryBackend {
    _db: sled::Db,
    batches: sled::Tree,
    index: EvidenceIndex,
    payload_root: PathBuf,
}

impl std::fmt::Debug for SledAmlMemoryBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SledAmlMemoryBackend")
            .field("persisted_batches", &self.batches.len())
            .field("payload_root", &self.payload_root)
            .finish()
    }
}

impl SledAmlMemoryBackend {
    /// Open or create a local store and rebuild the lexical index from verified
    /// persisted Add batches. Any malformed batch fails startup closed.
    pub fn open(
        store_path: impl AsRef<Path>,
        payload_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let db = sled::open(store_path.as_ref())
            .map_err(|error| format!("failed to open AML Sled store: {error}"))?;
        let batches = db
            .open_tree(ADD_BATCH_TREE)
            .map_err(|error| format!("failed to open AML Add batch tree: {error}"))?;
        let mut backend = Self {
            _db: db,
            batches,
            index: EvidenceIndex::new(),
            payload_root: payload_root.as_ref().to_path_buf(),
        };
        backend.rebuild_index()?;
        Ok(backend)
    }

    /// Number of durable Add batches currently retained. This is a count of
    /// request transactions, not a count of individual source messages.
    pub fn persisted_batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Remove stored Add batches older than the supplied Unix-millisecond
    /// cutoff, flush the deletions, and rebuild the index from what remains.
    /// Call with a cutoff no later than 30 days after an evaluation completes.
    pub fn purge_stored_before(&mut self, cutoff_unix_ms: u64) -> Result<usize, String> {
        let mut expired_keys = Vec::new();
        for item in self.batches.iter() {
            let (key, value) = item.map_err(|error| format!("failed to iterate AML store: {error}"))?;
            let batch = decode_and_validate_batch(&value)?;
            validate_batch_key(&key, &batch)?;
            if batch.stored_at_unix_ms < cutoff_unix_ms {
                expired_keys.push(key);
            }
        }

        for key in &expired_keys {
            self.batches
                .remove(key)
                .map_err(|error| format!("failed to purge expired AML batch: {error}"))?;
        }
        self.batches
            .flush()
            .map_err(|error| format!("failed to flush AML retention purge: {error}"))?;
        self.rebuild_index()?;
        Ok(expired_keys.len())
    }

    /// Explicit full-erasure operation for the end of an evaluation.
    pub fn purge_all(&mut self) -> Result<usize, String> {
        let removed = self.batches.len();
        self.batches
            .clear()
            .map_err(|error| format!("failed to clear AML evaluation data: {error}"))?;
        self.batches
            .flush()
            .map_err(|error| format!("failed to flush AML full purge: {error}"))?;
        self.index = EvidenceIndex::new();
        Ok(removed)
    }

    fn rebuild_index(&mut self) -> Result<(), String> {
        let mut rebuilt = EvidenceIndex::new();
        for item in self.batches.iter() {
            let (key, value) =
                item.map_err(|error| format!("failed to read AML persisted batch: {error}"))?;
            let batch = decode_and_validate_batch(&value)?;
            validate_batch_key(&key, &batch)?;
            let source_bytes = serde_json::to_vec(&batch.request)
                .map_err(|error| format!("failed to serialize persisted Add source: {error}"))?;
            rebuilt.insert_batch_atomic(batch.records, &source_bytes)?;
        }
        self.index = rebuilt;
        Ok(())
    }

    fn verify_batch_visible(
        index: &EvidenceIndex,
        batch: &PersistedAmlAddBatch,
        payload_root: &Path,
    ) -> Result<(), String> {
        let now_ms = current_unix_ms()?;
        for record in &batch.records {
            let EvidencePayload::Text { value } = &record.payload else {
                return Err("AML Text/Coding Add cannot index non-text evidence".into());
            };
            if !value.chars().any(char::is_alphanumeric) {
                return Err("AML lexical baseline requires at least one searchable alphanumeric token per message".into());
            }
            let hits = index.search_bm25(
                value,
                &batch.request.user_id,
                now_ms,
                payload_root,
                Bm25Config {
                    k1: 1.2,
                    b: 0.75,
                    max_items: usize::MAX,
                },
            )?;
            if !hits.iter().any(|hit| hit.evidence_id == record.evidence_id) {
                return Err(format!(
                    "Add evidence {:?} is not immediately retrievable by its indexed content",
                    record.evidence_id
                ));
            }
        }
        Ok(())
    }

    fn existing_batch(
        &self,
        request: &AmlAddRequest,
    ) -> Result<Option<PersistedAmlAddBatch>, String> {
        let key = request_key(&request.request_id);
        let Some(value) = self
            .batches
            .get(key)
            .map_err(|error| format!("failed to read AML idempotency key: {error}"))?
        else {
            return Ok(None);
        };
        let batch = decode_and_validate_batch(&value)?;
        if batch.request.request_id != request.request_id {
            return Err("AML idempotency key hash collision".into());
        }
        Ok(Some(batch))
    }

    fn receipt_from_existing(
        &mut self,
        request: &AmlAddRequest,
        existing: &PersistedAmlAddBatch,
        payload_sha256: &str,
    ) -> Result<AmlAddCommitReceipt, String> {
        let retry = classify_add_retry(
            Some(&AmlAddIdempotencyReceipt {
                request_id: existing.request.request_id.clone(),
                payload_sha256: existing.payload_sha256.clone(),
            }),
            request,
        )?;
        match retry {
            AmlAddRetryDecision::ReplayIdenticalRequest => {
                // A previous attempt may have written the CAS value but returned
                // an error during flush. Flush it again before treating this
                // retry as durable success. This also closes the CAS-loser path.
                self.batches
                    .flush()
                    .map_err(|error| format!("AML replay durability flush failed: {error}"))?;
                // Rebuild to pick up a durable write made by another backend
                // instance that won the compare-and-swap race.
                self.rebuild_index()?;
                Self::verify_batch_visible(&self.index, existing, &self.payload_root)?;
                Ok(AmlAddCommitReceipt {
                    request_id: request.request_id.clone(),
                    payload_sha256: payload_sha256.to_owned(),
                    durable: true,
                    searchable: true,
                })
            }
            AmlAddRetryDecision::ConflictRequestIdReusedWithDifferentPayload => {
                Err("request_id was already committed with a different Add payload".into())
            }
            AmlAddRetryDecision::NewRequest => {
                Err("existing idempotency record produced an impossible retry state".into())
            }
        }
    }
}

impl AmlMemoryBackend for SledAmlMemoryBackend {
    fn commit_add_and_confirm_searchable(
        &mut self,
        request: &AmlAddRequest,
        payload_sha256: &str,
    ) -> Result<AmlAddCommitReceipt, String> {
        request.validate()?;
        let expected_payload_sha256 = request.payload_sha256()?;
        if expected_payload_sha256 != payload_sha256 {
            return Err("caller payload digest does not match canonical Add request".into());
        }

        if let Some(existing) = self.existing_batch(request)? {
            return self.receipt_from_existing(request, &existing, payload_sha256);
        }

        let source_bytes = serde_json::to_vec(request)
            .map_err(|error| format!("failed to serialize Add source trajectory: {error}"))?;
        let records = build_evidence_records(request, payload_sha256)?;
        let mut candidate_index = self.index.clone();
        candidate_index.insert_batch_atomic(records.clone(), &source_bytes)?;

        let batch = PersistedAmlAddBatch {
            request: request.clone(),
            payload_sha256: payload_sha256.to_owned(),
            stored_at_unix_ms: current_unix_ms()?,
            records,
        };
        // Prove that every item is query-visible in the candidate index before
        // committing the durable batch. The success receipt follows only after
        // the same candidate has also been flushed to disk.
        Self::verify_batch_visible(&candidate_index, &batch, &self.payload_root)?;
        let encoded = serde_json::to_vec(&batch)
            .map_err(|error| format!("failed to serialize persisted Add batch: {error}"))?;
        let key = request_key(&request.request_id);

        match self
            .batches
            .compare_and_swap(key, None::<&[u8]>, Some(encoded.as_slice()))
            .map_err(|error| format!("AML Add compare-and-swap failed: {error}"))?
        {
            Ok(()) => {
                // Do not emit success until the write is flushed. The process-local
                // index is replaced only after the durable write succeeds.
                self.batches
                    .flush()
                    .map_err(|error| format!("AML Add durability flush failed: {error}"))?;
                self.index = candidate_index;
                Ok(AmlAddCommitReceipt {
                    request_id: request.request_id.clone(),
                    payload_sha256: payload_sha256.to_owned(),
                    durable: true,
                    searchable: true,
                })
            }
            Err(_cas_error) => {
                // Another writer may have committed the same request ID while
                // this candidate index was being prepared.
                let existing = self
                    .existing_batch(request)?
                    .ok_or_else(|| "Add idempotency compare-and-swap lost without a stored batch".to_string())?;
                self.receipt_from_existing(request, &existing, payload_sha256)
            }
        }
    }

    fn search_ranked(
        &self,
        request: &AmlSearchRequest,
    ) -> Result<AmlSearchResponse, String> {
        request.validate()?;
        let hits = self.index.search_bm25(
            &request.query,
            &request.user_id,
            current_unix_ms()?,
            &self.payload_root,
            Bm25Config {
                k1: 1.2,
                b: 0.75,
                max_items: request.top_k,
            },
        )?;
        let data = hits
            .into_iter()
            .map(|hit| AmlSearchResult {
                id: hit.evidence_id,
                content: hit.context.value,
                score: Some(hit.bm25_score),
                created_at: None,
            })
            .collect();
        Ok(AmlSearchResponse { data })
    }
}

fn build_evidence_records(
    request: &AmlAddRequest,
    payload_sha256: &str,
) -> Result<Vec<EvidenceRecord>, String> {
    let mut records = Vec::with_capacity(request.messages.len());
    for (index, message) in request.messages.iter().enumerate() {
        let message_timestamp = message
            .timestamp
            .map(|timestamp| {
                u64::try_from(timestamp)
                    .map_err(|_| format!("messages[{index}].timestamp must be non-negative"))
            })
            .transpose()?;
        if !message.content.chars().any(char::is_alphanumeric) {
            return Err(format!(
                "messages[{index}].content must contain a searchable alphanumeric token for the lexical baseline"
            ));
        }
        let identity = format!(
            "{}\0{}\0{}",
            request.user_id, request.request_id, index
        );
        let identity_digest = Sha256::digest(identity.as_bytes());
        let hdc_id = u64::from_be_bytes(
            identity_digest[..8]
                .try_into()
                .map_err(|_| "failed to derive stable HDC ID".to_string())?,
        );
        let evidence_id = format!("aml-{}", sha256_hex(identity.as_bytes()));
        let role_segment = match message.role {
            crate::memory_challenge_api::AmlMessageRole::User => "user",
            crate::memory_challenge_api::AmlMessageRole::Assistant => "assistant",
        };
        records.push(EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_owned(),
            evidence_id,
            hdc_id,
            scope_id: request.user_id.clone(),
            source: EvidenceSource {
                trajectory_id: request.session_id.clone(),
                step_id: Some(format!("{role_segment}-{index}")),
                step_index: Some(index as u64),
                source_sha256: payload_sha256.to_owned(),
                source_pointer: format!("/messages/{index}/content"),
            },
            payload: EvidencePayload::Text {
                value: message.content.clone(),
            },
            content_sha256: sha256_hex(message.content.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: message_timestamp,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        });
    }
    Ok(records)
}

fn validate_batch(batch: &PersistedAmlAddBatch) -> Result<(), String> {
    batch.request.validate()?;
    let digest = batch.request.payload_sha256()?;
    if digest != batch.payload_sha256 {
        return Err("persisted Add payload digest does not match request".into());
    }
    let expected_records = build_evidence_records(&batch.request, &batch.payload_sha256)?;
    if expected_records != batch.records {
        return Err("persisted evidence records do not match canonical Add request".into());
    }
    Ok(())
}

fn decode_and_validate_batch(bytes: &[u8]) -> Result<PersistedAmlAddBatch, String> {
    let batch: PersistedAmlAddBatch = serde_json::from_slice(bytes)
        .map_err(|error| format!("persisted AML Add batch is invalid JSON: {error}"))?;
    validate_batch(&batch)?;
    Ok(batch)
}

fn validate_batch_key(key: &[u8], batch: &PersistedAmlAddBatch) -> Result<(), String> {
    if key != request_key(&batch.request.request_id).as_slice() {
        return Err("persisted AML Add batch is stored under a non-canonical request key".into());
    }
    Ok(())
}

fn request_key(request_id: &str) -> Vec<u8> {
    let digest = sha256_hex(request_id.as_bytes());
    let mut key = Vec::with_capacity(ADD_KEY_PREFIX.len() + digest.len());
    key.extend_from_slice(ADD_KEY_PREFIX);
    key.extend_from_slice(digest.as_bytes());
    key
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn current_unix_ms() -> Result<u64, String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?;
    u64::try_from(elapsed.as_millis())
        .map_err(|_| "system timestamp exceeds u64 milliseconds".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_challenge_api::{
        AmlMessage, AmlMessageRole, execute_aml_add, execute_aml_search,
    };
    use tempfile::tempdir;

    fn add_request(request_id: &str, user_id: &str, session_id: &str, contents: &[&str]) -> AmlAddRequest {
        AmlAddRequest {
            request_id: request_id.into(),
            messages: contents
                .iter()
                .map(|content| AmlMessage {
                    role: AmlMessageRole::User,
                    content: (*content).into(),
                    timestamp: None,
                })
                .collect(),
            user_id: user_id.into(),
            session_id: session_id.into(),
        }
    }

    fn open_backend(root: &Path) -> SledAmlMemoryBackend {
        SledAmlMemoryBackend::open(root.join("db"), root).unwrap()
    }

    #[test]
    fn durable_add_is_immediately_searchable_and_survives_restart() {
        let root = tempdir().unwrap();
        {
            let mut backend = open_backend(root.path());
            let request = add_request(
                "request-1",
                "user-1",
                "session-1",
                &["Investigated stale cache key in checkout integration.", "The fix invalidated cache entries after schema migration."],
            );
            let response = execute_aml_add(&mut backend, request.clone()).unwrap();
            response.validate_echo(&request).unwrap();
            assert_eq!(backend.persisted_batch_count(), 1);
            let persisted = backend.existing_batch(&request).unwrap().unwrap();
            assert_eq!(persisted.records[0].source.step_id.as_deref(), Some("user-0"));
            assert_eq!(persisted.records[1].source.step_id.as_deref(), Some("assistant-1"));

            let result = execute_aml_search(
                &backend,
                AmlSearchRequest {
                    query: "stale cache key".into(),
                    options: None,
                    user_id: "user-1".into(),
                    top_k: 10,
                },
            ).unwrap();
            assert!(!result.data.is_empty());
            assert!(result.data[0].content.contains("stale cache key"));
        }

        let backend = open_backend(root.path());
        let result = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "schema migration".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
        ).unwrap();
        assert_eq!(result.data.len(), 1);
        assert!(result.data[0].content.contains("schema migration"));
    }

    #[test]
    fn add_is_idempotent_and_same_id_different_payload_is_rejected() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let request = add_request("request-1", "user-1", "session-1", &["Root cause was a stale cache key."]);
        execute_aml_add(&mut backend, request.clone()).unwrap();
        execute_aml_add(&mut backend, request.clone()).unwrap();
        assert_eq!(backend.persisted_batch_count(), 1);

        let changed = add_request("request-1", "user-1", "session-1", &["Different root cause and fix."]);
        assert!(execute_aml_add(&mut backend, changed).is_err());
        assert_eq!(backend.persisted_batch_count(), 1);
    }

    #[test]
    fn identical_retry_after_restart_flushes_and_reuses_existing_batch() {
        let root = tempdir().unwrap();
        let request = add_request(
            "request-1",
            "user-1",
            "session-1",
            &["Reproduced stale evidence after index restart."],
        );
        {
            let mut backend = open_backend(root.path());
            execute_aml_add(&mut backend, request.clone()).unwrap();
            assert_eq!(backend.persisted_batch_count(), 1);
        }
        {
            let mut reopened = open_backend(root.path());
            let response = execute_aml_add(&mut reopened, request.clone()).unwrap();
            response.validate_echo(&request).unwrap();
            assert_eq!(reopened.persisted_batch_count(), 1);
            let hits = execute_aml_search(&reopened, AmlSearchRequest {
                query: "stale evidence index restart".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            }).unwrap();
            assert_eq!(hits.data.len(), 1);
        }
    }

    #[test]
    fn user_id_is_the_only_scope_and_sessions_for_the_same_user_are_searchable() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let first = add_request("request-1", "user-1", "session-A", &["alpha repository cache resolver"]);
        let second = add_request("request-2", "user-1", "session-B", &["beta database migration runner"]);
        let other = add_request("request-3", "user-2", "session-A", &["gamma private deployment credentials"]);
        execute_aml_add(&mut backend, first).unwrap();
        execute_aml_add(&mut backend, second).unwrap();
        execute_aml_add(&mut backend, other).unwrap();

        let user_one = execute_aml_search(&backend, AmlSearchRequest {
            query: "repository beta database".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        assert_eq!(user_one.data.len(), 2);
        assert!(user_one.data.iter().all(|item| !item.content.contains("private deployment")));

        let user_two = execute_aml_search(&backend, AmlSearchRequest {
            query: "private deployment credentials".into(),
            options: None,
            user_id: "user-2".into(),
            top_k: 10,
        }).unwrap();
        assert_eq!(user_two.data.len(), 1);
        assert!(user_two.data[0].content.contains("private deployment"));
    }

    #[test]
    fn top_k_is_enforced_by_search_backend() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let request = add_request(
            "request-1",
            "user-1",
            "session-1",
            &["cache resolver route", "database cache resolver", "deployment cache resolver"],
        );
        execute_aml_add(&mut backend, request).unwrap();
        let result = execute_aml_search(&backend, AmlSearchRequest {
            query: "cache resolver".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 1,
        }).unwrap();
        assert!(result.data.len() <= 1);
    }

    #[test]
    fn cutoff_retention_purge_removes_old_batches_and_rebuilds_index() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request("request-1", "user-1", "session-1", &["old evaluation-only trajectory"]),
        ).unwrap();
        let cutoff = current_unix_ms().unwrap().saturating_add(1);
        assert_eq!(backend.purge_stored_before(cutoff).unwrap(), 1);
        assert_eq!(backend.persisted_batch_count(), 0);
        let result = execute_aml_search(&backend, AmlSearchRequest {
            query: "evaluation-only trajectory".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        assert!(result.data.is_empty());
    }

    #[test]
    fn retention_purge_removes_durable_batches_and_searchable_evidence() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request("request-1", "user-1", "session-1", &["evaluation-only private trajectory evidence"]),
        ).unwrap();
        assert_eq!(backend.persisted_batch_count(), 1);
        assert_eq!(backend.purge_all().unwrap(), 1);
        assert_eq!(backend.persisted_batch_count(), 0);
        let result = execute_aml_search(&backend, AmlSearchRequest {
            query: "private trajectory evidence".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        assert!(result.data.is_empty());
    }

    #[test]
    fn persisted_batch_under_noncanonical_key_fails_rebuild_closed() {
        let root = tempdir().unwrap();
        let path = root.path().join("db");
        let request = add_request("request-1", "user-1", "session-1", &["canonical stored content"]);
        let payload_sha256 = request.payload_sha256().unwrap();
        let records = build_evidence_records(&request, &payload_sha256).unwrap();
        let batch = PersistedAmlAddBatch {
            request,
            payload_sha256,
            stored_at_unix_ms: current_unix_ms().unwrap(),
            records,
        };
        {
            let db = sled::open(&path).unwrap();
            let tree = db.open_tree(ADD_BATCH_TREE).unwrap();
            tree.insert(b"wrong-key", serde_json::to_vec(&batch).unwrap()).unwrap();
            tree.flush().unwrap();
        }
        assert!(SledAmlMemoryBackend::open(&path, root.path())
            .unwrap_err()
            .contains("non-canonical request key"));
    }

    #[test]
    fn invalid_persisted_batch_fails_rebuild_closed() {
        let root = tempdir().unwrap();
        let path = root.path().join("db");
        {
            let db = sled::open(&path).unwrap();
            let tree = db.open_tree(ADD_BATCH_TREE).unwrap();
            tree.insert(request_key("malformed"), b"{not json").unwrap();
            tree.flush().unwrap();
        }
        assert!(SledAmlMemoryBackend::open(&path, root.path()).is_err());
    }
}
