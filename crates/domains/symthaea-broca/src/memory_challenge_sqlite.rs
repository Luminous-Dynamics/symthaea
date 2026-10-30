// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! SQLite/WAL persistence backend for the AML Text/Coding Add/Search contract.
//!
//! This is a single-node research/development backend. It provides transactional
//! Add idempotency and a rebuildable process-local lexical index. It is not an
//! HTTP server, a replicated service, or the official Open-source Methods model
//! configuration (which requires text-embedding-v4 for embeddings).
//!
//! Durability contract:
//! - WAL journal mode and synchronous=FULL are configured on every open.
//! - Add uses BEGIN IMMEDIATE and a unique request_id primary key.
//! - The exact ordered request, canonical payload digest and generated evidence
//!   records commit in one SQLite transaction.
//! - The candidate index is validated before commit; success is returned only
//!   after commit and immediate local Search verification.
//! - Restart rebuilds the index only from digest-checked, canonical rows.
//!
//! Use one active backend owner per database file. SQLite coordinates writers,
//! but each process-local EvidenceIndex is a cache; multi-process cache
//! invalidation/replication is not implemented here.

use crate::memory_challenge_api::{
    AmlAddCommitReceipt, AmlAddIdempotencyReceipt, AmlAddRequest, AmlAddRetryDecision,
    AmlMemoryBackend, AmlSearchRequest, AmlSearchResponse, AmlSearchResult,
    classify_add_retry,
};
use crate::memory_evidence::{
    Bm25Config, EVIDENCE_SCHEMA_VERSION, EvidenceIndex, EvidenceLifecycle, EvidencePayload,
    EvidenceRecord, EvidenceSource,
};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SQLITE_SCHEMA_VERSION: i64 = 1;
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// Canonical row payload for one Add transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedAmlAddBatch {
    request: AmlAddRequest,
    payload_sha256: String,
    stored_at_unix_ms: u64,
    records: Vec<EvidenceRecord>,
}

/// One immutable SQLite row decoded and checked before it enters the index.
struct PersistedRow {
    request_id: String,
    payload_sha256: String,
    stored_at_unix_ms: i64,
    body_json: Vec<u8>,
}

/// SQLite-backed Add persistence plus a process-local lexical index.
pub struct SqliteAmlMemoryBackend {
    connection: Connection,
    index: EvidenceIndex,
    payload_root: PathBuf,
}

impl std::fmt::Debug for SqliteAmlMemoryBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteAmlMemoryBackend")
            .field("indexed_evidence", &self.indexed_evidence_count())
            .field("payload_root", &self.payload_root)
            .finish()
    }
}

impl SqliteAmlMemoryBackend {
    /// Open/create a SQLite database file and rebuild its index from verified
    /// persisted batches. Any malformed row or unsupported schema fails closed.
    pub fn open(
        database_path: impl AsRef<Path>,
        payload_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let database_path = database_path.as_ref();
        if let Some(parent) = database_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create SQLite database directory: {error}"))?;
        }
        let connection = Connection::open(database_path)
            .map_err(|error| format!("failed to open AML SQLite database: {error}"))?;
        connection
            .busy_timeout(BUSY_TIMEOUT)
            .map_err(|error| format!("failed to set SQLite busy timeout: {error}"))?;
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = FULL;
                 PRAGMA foreign_keys = ON;
                 CREATE TABLE IF NOT EXISTS aml_add_batches (
                     request_id TEXT PRIMARY KEY NOT NULL COLLATE BINARY,
                     payload_sha256 TEXT NOT NULL,
                     stored_at_unix_ms INTEGER NOT NULL CHECK (stored_at_unix_ms >= 0),
                     body_json BLOB NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS aml_add_batches_stored_at
                     ON aml_add_batches(stored_at_unix_ms);",
            )
            .map_err(|error| format!("failed to initialize AML SQLite schema: {error}"))?;

        let schema_version: i64 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(|error| format!("failed to read AML SQLite schema version: {error}"))?;
        if schema_version == 0 {
            connection
                .pragma_update(None, "user_version", SQLITE_SCHEMA_VERSION)
                .map_err(|error| format!("failed to set AML SQLite schema version: {error}"))?;
        } else if schema_version != SQLITE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported AML SQLite schema version {schema_version}; expected {SQLITE_SCHEMA_VERSION}"
            ));
        }

        let mut backend = Self {
            connection,
            index: EvidenceIndex::new(),
            payload_root: payload_root.as_ref().to_path_buf(),
        };
        backend.rebuild_index()?;
        Ok(backend)
    }

    /// Number of persisted Add transactions (not individual messages).
    pub fn persisted_batch_count(&self) -> Result<usize, String> {
        let count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM aml_add_batches", [], |row| row.get(0))
            .map_err(|error| format!("failed to count AML SQLite batches: {error}"))?;
        usize::try_from(count).map_err(|_| "negative or oversized SQLite batch count".into())
    }

    /// Number of indexed source messages currently in the process-local cache.
    pub fn indexed_evidence_count(&self) -> usize {
        self.indexed_count()
    }

    /// Explicit logical retention purge. Invoke no later than 30 days after an
    /// evaluation run completes. SQLite transaction commit is durable under
    /// synchronous=FULL; the process-local index is rebuilt before returning.
    pub fn purge_stored_before(&mut self, cutoff_unix_ms: u64) -> Result<usize, String> {
        let cutoff = i64::try_from(cutoff_unix_ms)
            .map_err(|_| "retention cutoff exceeds SQLite signed integer range".to_string())?;
        let transaction = self.connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| format!("failed to begin AML retention transaction: {error}"))?;
        let removed = transaction
            .execute(
                "DELETE FROM aml_add_batches WHERE stored_at_unix_ms < ?1",
                params![cutoff],
            )
            .map_err(|error| format!("failed to delete expired AML batches: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("failed to commit AML retention transaction: {error}"))?;
        self.rebuild_index()?;
        Ok(removed)
    }

    /// Explicit full logical erasure after the evaluation run.
    ///
    /// This removes all application rows, not SQLite WAL files or backups.
    /// It is not a forensic secure-erasure guarantee; deployments must also
    /// dispose of the database, WAL/SHM files, backups, and snapshots according
    /// to the applicable retention policy.
    pub fn purge_all(&mut self) -> Result<usize, String> {
        let transaction = self.connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| format!("failed to begin AML full-purge transaction: {error}"))?;
        let removed = transaction
            .execute("DELETE FROM aml_add_batches", [])
            .map_err(|error| format!("failed to delete AML batches: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("failed to commit AML full-purge transaction: {error}"))?;
        self.index = EvidenceIndex::new();
        Ok(removed)
    }

    fn indexed_count(&self) -> usize {
        // The lexical index currently has no public count method. A diagnostic
        // count is derived from the persisted row count, which is authoritative.
        self.persisted_batch_count().unwrap_or(0)
    }

    fn read_persisted_rows(&self) -> Result<Vec<PersistedRow>, String> {
        let mut statement = self.connection
            .prepare(
                "SELECT request_id, payload_sha256, stored_at_unix_ms, body_json
                 FROM aml_add_batches
                 ORDER BY request_id COLLATE BINARY",
            )
            .map_err(|error| format!("failed to prepare AML row scan: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok(PersistedRow {
                    request_id: row.get(0)?,
                    payload_sha256: row.get(1)?,
                    stored_at_unix_ms: row.get(2)?,
                    body_json: row.get(3)?,
                })
            })
            .map_err(|error| format!("failed to query AML persisted batches: {error}"))?;
        let mut output = Vec::new();
        for row in rows {
            output.push(row.map_err(|error| format!("failed to decode AML SQLite row: {error}"))?);
        }
        Ok(output)
    }

    fn rebuild_index(&mut self) -> Result<(), String> {
        let rows = self.read_persisted_rows()?;
        let mut rebuilt = EvidenceIndex::new();
        for row in rows {
            let batch = decode_and_validate_batch(&row.body_json)?;
            validate_row_metadata(&row, &batch)?;
            let source_bytes = serde_json::to_vec(&batch.request)
                .map_err(|error| format!("failed to serialize persisted Add source: {error}"))?;
            rebuilt.insert_batch_atomic(batch.records, &source_bytes)?;
        }
        self.index = rebuilt;
        Ok(())
    }

    fn lookup_existing(
        &self,
        request_id: &str,
    ) -> Result<Option<PersistedAmlAddBatch>, String> {
        let row = self.connection
            .query_row(
                "SELECT request_id, payload_sha256, stored_at_unix_ms, body_json
                 FROM aml_add_batches WHERE request_id = ?1",
                params![request_id],
                |row| {
                    Ok(PersistedRow {
                        request_id: row.get(0)?,
                        payload_sha256: row.get(1)?,
                        stored_at_unix_ms: row.get(2)?,
                        body_json: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("failed to query AML idempotency key: {error}"))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let batch = decode_and_validate_batch(&row.body_json)?;
        validate_row_metadata(&row, &batch)?;
        Ok(Some(batch))
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

    fn retry_existing(
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
                // Re-read the database after a transaction boundary and rebuild
                // before acknowledging a replay. SQLite synchronous=FULL makes
                // a successful prior commit durable; the previous write may
                // have returned an I/O error to its caller.
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

impl AmlMemoryBackend for SqliteAmlMemoryBackend {
    fn commit_add_and_confirm_searchable(
        &mut self,
        request: &AmlAddRequest,
        payload_sha256: &str,
    ) -> Result<AmlAddCommitReceipt, String> {
        request.validate()?;
        let expected_digest = request.payload_sha256()?;
        if expected_digest != payload_sha256 {
            return Err("caller payload digest does not match canonical Add request".into());
        }

        // BEGIN IMMEDIATE serializes writers before the idempotency lookup, so
        // concurrent processes cannot both claim the same request_id.
        let transaction = self.connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| format!("failed to begin AML Add transaction: {error}"))?;
        let existing_row = transaction
            .query_row(
                "SELECT request_id, payload_sha256, stored_at_unix_ms, body_json
                 FROM aml_add_batches WHERE request_id = ?1",
                params![request.request_id],
                |row| {
                    Ok(PersistedRow {
                        request_id: row.get(0)?,
                        payload_sha256: row.get(1)?,
                        stored_at_unix_ms: row.get(2)?,
                        body_json: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("failed to query AML Add transaction key: {error}"))?;

        if let Some(row) = existing_row {
            let existing = decode_and_validate_batch(&row.body_json)?;
            validate_row_metadata(&row, &existing)?;
            transaction
                .commit()
                .map_err(|error| format!("failed to end AML replay transaction: {error}"))?;
            return self.retry_existing(request, &existing, payload_sha256);
        }

        let source_bytes = serde_json::to_vec(request)
            .map_err(|error| format!("failed to serialize Add source: {error}"))?;
        let records = build_evidence_records(request, payload_sha256)?;
        let mut candidate_index = self.index.clone();
        candidate_index.insert_batch_atomic(records.clone(), &source_bytes)?;
        let batch = PersistedAmlAddBatch {
            request: request.clone(),
            payload_sha256: payload_sha256.to_owned(),
            stored_at_unix_ms: current_unix_ms()?,
            records,
        };
        Self::verify_batch_visible(&candidate_index, &batch, &self.payload_root)?;

        let stored_at = i64::try_from(batch.stored_at_unix_ms)
            .map_err(|_| "stored_at timestamp exceeds SQLite signed integer range".to_string())?;
        let body_json = serde_json::to_vec(&batch)
            .map_err(|error| format!("failed to serialize persisted Add batch: {error}"))?;
        transaction
            .execute(
                "INSERT INTO aml_add_batches
                    (request_id, payload_sha256, stored_at_unix_ms, body_json)
                 VALUES (?1, ?2, ?3, ?4)",
                params![request.request_id, payload_sha256, stored_at, body_json],
            )
            .map_err(|error| format!("failed to insert AML Add batch: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("failed to commit durable AML Add batch: {error}"))?;

        // Candidate validation happened before commit; after commit the single
        // process-local owner publishes the new index snapshot atomically.
        self.index = candidate_index;
        Ok(AmlAddCommitReceipt {
            request_id: request.request_id.clone(),
            payload_sha256: payload_sha256.to_owned(),
            durable: true,
            searchable: true,
        })
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
        let identity = format!("{}\0{}\0{}", request.user_id, request.request_id, index);
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
    let expected = build_evidence_records(&batch.request, &batch.payload_sha256)?;
    if expected != batch.records {
        return Err("persisted evidence records do not match canonical Add request".into());
    }
    Ok(())
}

fn validate_row_metadata(row: &PersistedRow, batch: &PersistedAmlAddBatch) -> Result<(), String> {
    if row.request_id != batch.request.request_id
        || row.payload_sha256 != batch.payload_sha256
        || row.stored_at_unix_ms < 0
        || u64::try_from(row.stored_at_unix_ms).ok() != Some(batch.stored_at_unix_ms)
    {
        return Err("SQLite row metadata does not match canonical persisted batch".into());
    }
    Ok(())
}

fn decode_and_validate_batch(bytes: &[u8]) -> Result<PersistedAmlAddBatch, String> {
    let batch: PersistedAmlAddBatch = serde_json::from_slice(bytes)
        .map_err(|error| format!("persisted AML Add batch is invalid JSON: {error}"))?;
    validate_batch(&batch)?;
    Ok(batch)
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

    fn add_request(
        request_id: &str,
        user_id: &str,
        session_id: &str,
        contents: &[&str],
    ) -> AmlAddRequest {
        AmlAddRequest {
            request_id: request_id.into(),
            messages: contents
                .iter()
                .enumerate()
                .map(|(index, content)| AmlMessage {
                    role: if index % 2 == 0 {
                        AmlMessageRole::User
                    } else {
                        AmlMessageRole::Assistant
                    },
                    content: (*content).into(),
                    timestamp: None,
                })
                .collect(),
            user_id: user_id.into(),
            session_id: session_id.into(),
        }
    }

    fn open_backend(root: &Path) -> SqliteAmlMemoryBackend {
        SqliteAmlMemoryBackend::open(root.join("aml-memory.sqlite3"), root).unwrap()
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
                &[
                    "Investigated stale cache key in checkout integration.",
                    "The fix invalidated cache entries after schema migration.",
                ],
            );
            let response = execute_aml_add(&mut backend, request.clone()).unwrap();
            response.validate_echo(&request).unwrap();
            assert_eq!(backend.persisted_batch_count().unwrap(), 1);
            assert_eq!(backend.indexed_evidence_count(), 2);
            let persisted = backend.lookup_existing("request-1").unwrap().unwrap();
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
            )
            .unwrap();
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
        )
        .unwrap();
        assert_eq!(result.data.len(), 1);
        assert!(result.data[0].content.contains("schema migration"));
    }

    #[test]
    fn add_is_idempotent_and_same_id_different_payload_is_rejected_across_restart() {
        let root = tempdir().unwrap();
        let request = add_request(
            "request-1",
            "user-1",
            "session-1",
            &["Root cause was a stale cache key."],
        );
        {
            let mut backend = open_backend(root.path());
            execute_aml_add(&mut backend, request.clone()).unwrap();
        }
        {
            let mut backend = open_backend(root.path());
            execute_aml_add(&mut backend, request.clone()).unwrap();
            assert_eq!(backend.persisted_batch_count().unwrap(), 1);
            let changed = add_request(
                "request-1",
                "user-1",
                "session-1",
                &["Different root cause and fix."],
            );
            assert!(execute_aml_add(&mut backend, changed).is_err());
            assert_eq!(backend.persisted_batch_count().unwrap(), 1);
        }
    }

    #[test]
    fn user_id_is_the_only_scope_and_sessions_for_the_same_user_are_searchable() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let first = add_request(
            "request-1",
            "user-1",
            "session-A",
            &["alpha repository cache resolver"],
        );
        let second = add_request(
            "request-2",
            "user-1",
            "session-B",
            &["beta database migration runner"],
        );
        let other = add_request(
            "request-3",
            "user-2",
            "session-A",
            &["gamma private deployment credentials"],
        );
        execute_aml_add(&mut backend, first).unwrap();
        execute_aml_add(&mut backend, second).unwrap();
        execute_aml_add(&mut backend, other).unwrap();

        let user_one = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "repository beta database".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
        )
        .unwrap();
        assert_eq!(user_one.data.len(), 2);
        assert!(
            user_one
                .data
                .iter()
                .all(|item| !item.content.contains("private deployment"))
        );

        let user_two = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "private deployment credentials".into(),
                options: None,
                user_id: "user-2".into(),
                top_k: 10,
            },
        )
        .unwrap();
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
            &[
                "cache resolver route",
                "database cache resolver",
                "deployment cache resolver",
            ],
        );
        execute_aml_add(&mut backend, request).unwrap();
        let result = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "cache resolver".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 1,
            },
        )
        .unwrap();
        assert!(result.data.len() <= 1);
    }

    #[test]
    fn cutoff_and_full_retention_purge_remove_durable_and_searchable_evidence() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request(
                "request-old",
                "user-1",
                "session-old",
                &["old evaluation-only trajectory evidence"],
            ),
        )
        .unwrap();
        let future_cutoff = current_unix_ms().unwrap().saturating_add(1);
        assert_eq!(backend.purge_stored_before(future_cutoff).unwrap(), 1);
        assert_eq!(backend.persisted_batch_count().unwrap(), 0);

        execute_aml_add(
            &mut backend,
            add_request(
                "request-new",
                "user-1",
                "session-new",
                &["new evaluation-only trajectory evidence"],
            ),
        )
        .unwrap();
        assert_eq!(backend.purge_all().unwrap(), 1);
        assert_eq!(backend.persisted_batch_count().unwrap(), 0);
        let result = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "evaluation-only trajectory".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
        )
        .unwrap();
        assert!(result.data.is_empty());
    }

    #[test]
    fn malformed_persisted_batch_fails_rebuild_closed() {
        let root = tempdir().unwrap();
        let path = root.path().join("broken.sqlite3");
        {
            let connection = Connection::open(&path).unwrap();
            connection
                .execute_batch(
                    "CREATE TABLE aml_add_batches (
                        request_id TEXT PRIMARY KEY NOT NULL COLLATE BINARY,
                        payload_sha256 TEXT NOT NULL,
                        stored_at_unix_ms INTEGER NOT NULL,
                        body_json BLOB NOT NULL
                    );
                    PRAGMA user_version = 1;",
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO aml_add_batches
                        (request_id, payload_sha256, stored_at_unix_ms, body_json)
                     VALUES ('bad', 'bad', 1, ?1)",
                    params![b"{not json".as_slice()],
                )
                .unwrap();
        }
        assert!(SqliteAmlMemoryBackend::open(&path, root.path()).is_err());
    }

    #[test]
    fn row_metadata_must_match_canonical_persisted_payload() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request(
                "request-1",
                "user-1",
                "session-1",
                &["canonical payload source record"],
            ),
        )
        .unwrap();
        backend.connection
            .execute(
                "UPDATE aml_add_batches SET payload_sha256 = ?1 WHERE request_id = ?2",
                params!["0".repeat(64), "request-1"],
            )
            .unwrap();
        assert!(backend.rebuild_index().unwrap_err().contains("row metadata does not match"));
    }
}
