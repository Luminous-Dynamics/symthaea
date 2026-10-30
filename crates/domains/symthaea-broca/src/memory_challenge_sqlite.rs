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
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SQLITE_SCHEMA_VERSION: i64 = 3;
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

/// Canonical row payload for one Add transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedAmlAddBatch {
    request: AmlAddRequest,
    payload_sha256: String,
    stored_at_unix_ms: u64,
    records: Vec<EvidenceRecord>,
}

/// Append-only lifecycle transition over one or two immutable evidence IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum EvidenceLifecycleOperation {
    Revoke { evidence_id: String },
    Delete { evidence_id: String },
    Supersede {
        old_evidence_id: String,
        new_evidence_id: String,
    },
}

impl EvidenceLifecycleOperation {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Revoke { evidence_id } | Self::Delete { evidence_id } => {
                require_non_empty(evidence_id, "evidence_id")
            }
            Self::Supersede { old_evidence_id, new_evidence_id } => {
                require_non_empty(old_evidence_id, "old_evidence_id")?;
                require_non_empty(new_evidence_id, "new_evidence_id")?;
                if old_evidence_id == new_evidence_id {
                    return Err("old_evidence_id and new_evidence_id must differ".into());
                }
                Ok(())
            }
        }
    }

    fn references_any(&self, evidence_ids: &BTreeSet<String>) -> bool {
        match self {
            Self::Revoke { evidence_id } | Self::Delete { evidence_id } => {
                evidence_ids.contains(evidence_id)
            }
            Self::Supersede { old_evidence_id, new_evidence_id } => {
                evidence_ids.contains(old_evidence_id) || evidence_ids.contains(new_evidence_id)
            }
        }
    }

    fn apply(&self, index: &mut EvidenceIndex) -> Result<(), String> {
        match self {
            Self::Revoke { evidence_id } => index.revoke_evidence(evidence_id),
            Self::Delete { evidence_id } => index.delete_evidence(evidence_id),
            Self::Supersede { old_evidence_id, new_evidence_id } => {
                index.supersede_evidence(old_evidence_id, new_evidence_id)
            }
        }
    }
}

/// Immutable lifecycle event. sequence is assigned by SQLite and defines
/// replay order; event_id makes caller retries idempotent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedEvidenceLifecycleEvent {
    event_id: String,
    recorded_at_unix_ms: u64,
    operation: EvidenceLifecycleOperation,
}

struct PersistedLifecycleRow {
    sequence: i64,
    event_id: String,
    recorded_at_unix_ms: i64,
    event_sha256: String,
    body_json: Vec<u8>,
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
            .field("persisted_batches", &self.persisted_batch_count())
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
        let mut connection = Connection::open(database_path)
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
                     ON aml_add_batches(stored_at_unix_ms);
                 CREATE TABLE IF NOT EXISTS aml_evidence_lifecycle_events (
                     sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                     event_id TEXT UNIQUE NOT NULL COLLATE BINARY,
                     recorded_at_unix_ms INTEGER NOT NULL CHECK (recorded_at_unix_ms >= 0),
                     event_sha256 TEXT NOT NULL,
                     body_json BLOB NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS aml_evidence_lifecycle_events_recorded_at
                     ON aml_evidence_lifecycle_events(recorded_at_unix_ms);",
            )
            .map_err(|error| format!("failed to initialize AML SQLite schema: {error}"))?;

        let journal_mode: String = connection
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .map_err(|error| format!("failed to verify SQLite journal mode: {error}"))?;
        if !journal_mode.eq_ignore_ascii_case("wal") {
            return Err(format!(
                "AML SQLite backend requires WAL journal mode, got {journal_mode:?}"
            ));
        }
        let synchronous: i64 = connection
            .pragma_query_value(None, "synchronous", |row| row.get(0))
            .map_err(|error| format!("failed to verify SQLite synchronous mode: {error}"))?;
        if synchronous != 2 {
            return Err(format!(
                "AML SQLite backend requires synchronous=FULL (2), got {synchronous}"
            ));
        }
        let integrity: String = connection
            .query_row("PRAGMA quick_check(1)", [], |row| row.get(0))
            .map_err(|error| format!("SQLite quick_check failed to execute: {error}"))?;
        if integrity != "ok" {
            return Err(format!("SQLite quick_check reported database issue: {integrity}"));
        }

        let schema_version: i64 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(|error| format!("failed to read AML SQLite schema version: {error}"))?;
        match schema_version {
            0 | 1 => {
                // Fresh databases and v1 Add-only stores get the v3 lifecycle
                // schema created above. Existing Add batches are untouched.
                connection
                    .pragma_update(None, "user_version", SQLITE_SCHEMA_VERSION)
                    .map_err(|error| format!("failed to set AML SQLite schema version: {error}"))?;
            }
            2 => {
                // v2 lifecycle rows predate the content-digest column. Migrate
                // atomically: hash each exact stored body, then advance version.
                let transaction = connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(|error| format!("failed to begin SQLite v2->v3 migration: {error}"))?;
                transaction
                    .execute_batch(
                        "ALTER TABLE aml_evidence_lifecycle_events
                         ADD COLUMN event_sha256 TEXT NOT NULL DEFAULT '';"
                    )
                    .map_err(|error| format!("failed to add lifecycle event digest column: {error}"))?;
                let rows: Vec<(String, Vec<u8>)> = {
                    let mut statement = transaction
                        .prepare("SELECT event_id, body_json FROM aml_evidence_lifecycle_events ORDER BY sequence ASC")
                        .map_err(|error| format!("failed to read v2 lifecycle rows: {error}"))?;
                    let mapped = statement
                        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                        .map_err(|error| format!("failed to query v2 lifecycle rows: {error}"))?;
                    let mut rows = Vec::new();
                    for row in mapped {
                        rows.push(row.map_err(|error| format!("failed to decode v2 lifecycle row: {error}"))?);
                    }
                    rows
                };
                for (event_id, body_json) in rows {
                    let event: PersistedEvidenceLifecycleEvent = serde_json::from_slice(&body_json)
                        .map_err(|error| format!("v2 lifecycle event {event_id:?} is invalid: {error}"))?;
                    require_non_empty(&event.event_id, "lifecycle event_id")?;
                    event.operation.validate()?;
                    if event.event_id != event_id {
                        return Err("v2 lifecycle row event_id does not match its payload".into());
                    }
                    let digest = sha256_hex(&body_json);
                    transaction
                        .execute(
                            "UPDATE aml_evidence_lifecycle_events SET event_sha256 = ?1 WHERE event_id = ?2",
                            params![digest, event_id],
                        )
                        .map_err(|error| format!("failed to backfill v2 lifecycle digest: {error}"))?;
                }
                transaction
                    .execute_batch("PRAGMA user_version = 3;")
                    .map_err(|error| format!("failed to advance SQLite schema to v3: {error}"))?;
                transaction
                    .commit()
                    .map_err(|error| format!("failed to commit SQLite v2->v3 migration: {error}"))?;
            }
            version if version == SQLITE_SCHEMA_VERSION => {}
            version => {
                return Err(format!(
                    "unsupported AML SQLite schema version {version}; expected {SQLITE_SCHEMA_VERSION}"
                ));
            }
        }

        let mut backend = Self {
            connection,
            index: EvidenceIndex::new(),
            payload_root: payload_root.as_ref().to_path_buf(),
        };
        backend.rebuild_index()?;
        Ok(backend)
    }

    /// Reconstruct memory exactly as it was known at the supplied transaction
    /// time, then query it at the separate world-valid time.
    ///
    /// Add batches are visible only when their commit timestamp is <= known_at.
    /// Revoke/delete/supersede events are replayed only when their recorded_at
    /// timestamp is <= known_at. EvidenceRecord valid-time bounds are evaluated
    /// at valid_at. This method is the explicit (valid_at, known_at) query path;
    /// the public AML Search ABI continues to request the current view only.
    pub fn search_as_known_at(
        &self,
        request: &AmlSearchRequest,
        valid_at_unix_ms: u64,
        known_at_unix_ms: u64,
    ) -> Result<AmlSearchResponse, String> {
        request.validate()?;
        let index = self.build_index_as_known_at(known_at_unix_ms)?;
        Self::search_index(
            &index,
            request,
            valid_at_unix_ms,
            &self.payload_root,
        )
    }

    /// Number of persisted Add transactions (not individual messages).
    pub fn persisted_batch_count(&self) -> Result<usize, String> {
        let count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM aml_add_batches", [], |row| row.get(0))
            .map_err(|error| format!("failed to count AML SQLite batches: {error}"))?;
        usize::try_from(count).map_err(|_| "negative or oversized SQLite batch count".into())
    }

    /// Explicit logical retention purge. Invoke no later than 30 days after an
    /// evaluation run completes. SQLite transaction commit is durable under
    /// synchronous=FULL; the process-local index is rebuilt before returning.
    pub fn purge_stored_before(&mut self, cutoff_unix_ms: u64) -> Result<usize, String> {
        let cutoff = i64::try_from(cutoff_unix_ms)
            .map_err(|_| "retention cutoff exceeds SQLite signed integer range".to_string())?;

        // Read/validate the rows before opening the mutable transaction so
        // expired source IDs and every dependent lifecycle event are known.
        let mut expired_evidence_ids = BTreeSet::new();
        for row in self.read_persisted_rows()? {
            if row.stored_at_unix_ms < cutoff {
                let batch = decode_and_validate_batch(&row.body_json)?;
                validate_row_metadata(&row, &batch)?;
                expired_evidence_ids.extend(
                    batch.records.iter().map(|record| record.evidence_id.clone()),
                );
            }
        }
        let mut expired_event_ids = Vec::new();
        for row in self.read_persisted_lifecycle_rows()? {
            let event = decode_and_validate_lifecycle_row(&row)?;
            if row.recorded_at_unix_ms < cutoff
                || event.operation.references_any(&expired_evidence_ids)
            {
                expired_event_ids.push(row.event_id);
            }
        }

        let transaction = self.connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| format!("failed to begin AML retention transaction: {error}"))?;
        let removed = transaction
            .execute(
                "DELETE FROM aml_add_batches WHERE stored_at_unix_ms < ?1",
                params![cutoff],
            )
            .map_err(|error| format!("failed to delete expired AML batches: {error}"))?;
        for event_id in expired_event_ids {
            transaction
                .execute(
                    "DELETE FROM aml_evidence_lifecycle_events WHERE event_id = ?1",
                    params![event_id],
                )
                .map_err(|error| format!("failed to purge expired lifecycle event: {error}"))?;
        }
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
            .execute("DELETE FROM aml_evidence_lifecycle_events", [])
            .map_err(|error| format!("failed to delete AML lifecycle events: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("failed to commit AML full-purge transaction: {error}"))?;
        self.index = EvidenceIndex::new();
        Ok(removed)
    }

    /// Number of lifecycle transitions retained in the append-only event log.
    pub fn lifecycle_event_count(&self) -> Result<usize, String> {
        let count: i64 = self.connection
            .query_row("SELECT COUNT(*) FROM aml_evidence_lifecycle_events", [], |row| row.get(0))
            .map_err(|error| format!("failed to count AML lifecycle events: {error}"))?;
        usize::try_from(count).map_err(|_| "negative or oversized lifecycle event count".into())
    }

    /// Persist and apply a source retraction. Repeating the same event_id and
    /// operation is an idempotent replay; reusing the ID for a different
    /// operation is rejected.
    pub fn revoke_evidence(
        &mut self,
        event_id: &str,
        evidence_id: &str,
    ) -> Result<(), String> {
        self.append_lifecycle_event(
            event_id,
            EvidenceLifecycleOperation::Revoke { evidence_id: evidence_id.into() },
        )
    }

    /// Persist and apply an active-index deletion tombstone. The event remains
    /// auditable and is replayed before Search after restart.
    pub fn delete_evidence(
        &mut self,
        event_id: &str,
        evidence_id: &str,
    ) -> Result<(), String> {
        self.append_lifecycle_event(
            event_id,
            EvidenceLifecycleOperation::Delete { evidence_id: evidence_id.into() },
        )
    }

    /// Persist and apply an explicit supersession relation. Both source records
    /// must exist, be active, and satisfy EvidenceIndex's scope constraints.
    pub fn supersede_evidence(
        &mut self,
        event_id: &str,
        old_evidence_id: &str,
        new_evidence_id: &str,
    ) -> Result<(), String> {
        self.append_lifecycle_event(
            event_id,
            EvidenceLifecycleOperation::Supersede {
                old_evidence_id: old_evidence_id.into(),
                new_evidence_id: new_evidence_id.into(),
            },
        )
    }

    fn append_lifecycle_event(
        &mut self,
        event_id: &str,
        operation: EvidenceLifecycleOperation,
    ) -> Result<(), String> {
        require_non_empty(event_id, "lifecycle event_id")?;
        operation.validate()?;
        let transaction = self.connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| format!("failed to begin AML lifecycle transaction: {error}"))?;
        let existing = transaction
            .query_row(
                "SELECT sequence, event_id, recorded_at_unix_ms, event_sha256, body_json
                 FROM aml_evidence_lifecycle_events WHERE event_id = ?1",
                params![event_id],
                |row| Ok(PersistedLifecycleRow {
                    sequence: row.get(0)?,
                    event_id: row.get(1)?,
                    recorded_at_unix_ms: row.get(2)?,
                    event_sha256: row.get(3)?,
                    body_json: row.get(4)?,
                }),
            )
            .optional()
            .map_err(|error| format!("failed to read AML lifecycle idempotency key: {error}"))?;

        if let Some(row) = existing {
            let existing_event = decode_and_validate_lifecycle_row(&row)?;
            if existing_event.operation != operation {
                return Err("lifecycle event_id was reused for a different operation".into());
            }
            transaction
                .commit()
                .map_err(|error| format!("failed to end AML lifecycle replay transaction: {error}"))?;
            self.rebuild_index()?;
            return Ok(());
        }

        let mut candidate_index = self.index.clone();
        operation.apply(&mut candidate_index)?;
        let event = PersistedEvidenceLifecycleEvent {
            event_id: event_id.into(),
            recorded_at_unix_ms: next_ledger_transaction_time_unix_ms(&transaction)?,
            operation,
        };
        let recorded_at = i64::try_from(event.recorded_at_unix_ms)
            .map_err(|_| "lifecycle timestamp exceeds SQLite signed integer range".to_string())?;
        let body_json = serde_json::to_vec(&event)
            .map_err(|error| format!("failed to serialize AML lifecycle event: {error}"))?;
        let event_sha256 = sha256_hex(&body_json);
        transaction
            .execute(
                "INSERT INTO aml_evidence_lifecycle_events
                    (event_id, recorded_at_unix_ms, event_sha256, body_json)
                 VALUES (?1, ?2, ?3, ?4)",
                params![event.event_id, recorded_at, event_sha256, body_json],
            )
            .map_err(|error| format!("failed to persist AML lifecycle event: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("failed to commit AML lifecycle event: {error}"))?;
        self.index = candidate_index;
        Ok(())
    }

    fn read_persisted_lifecycle_rows(&self) -> Result<Vec<PersistedLifecycleRow>, String> {
        let mut statement = self.connection
            .prepare(
                "SELECT sequence, event_id, recorded_at_unix_ms, event_sha256, body_json
                 FROM aml_evidence_lifecycle_events
                 ORDER BY sequence ASC",
            )
            .map_err(|error| format!("failed to prepare AML lifecycle scan: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok(PersistedLifecycleRow {
                    sequence: row.get(0)?,
                    event_id: row.get(1)?,
                    recorded_at_unix_ms: row.get(2)?,
                    event_sha256: row.get(3)?,
                    body_json: row.get(4)?,
                })
            })
            .map_err(|error| format!("failed to query AML lifecycle events: {error}"))?;
        let mut output = Vec::new();
        for row in rows {
            output.push(row.map_err(|error| format!("failed to decode AML lifecycle row: {error}"))?);
        }
        Ok(output)
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
        self.index = self.build_index_as_known_at(u64::MAX)?;
        Ok(())
    }

    fn build_index_as_known_at(&self, known_at_unix_ms: u64) -> Result<EvidenceIndex, String> {
        let mut rebuilt = EvidenceIndex::new();
        for row in self.read_persisted_rows()? {
            if row.stored_at_unix_ms < 0
                || u64::try_from(row.stored_at_unix_ms).ok().is_none_or(|stored| stored > known_at_unix_ms)
            {
                continue;
            }
            let batch = decode_and_validate_batch(&row.body_json)?;
            validate_row_metadata(&row, &batch)?;
            let source_bytes = serde_json::to_vec(&batch.request)
                .map_err(|error| format!("failed to serialize persisted Add source: {error}"))?;
            rebuilt.insert_batch_atomic(batch.records, &source_bytes)?;
        }
        for row in self.read_persisted_lifecycle_rows()? {
            if row.recorded_at_unix_ms < 0
                || u64::try_from(row.recorded_at_unix_ms).ok().is_none_or(|recorded| recorded > known_at_unix_ms)
            {
                continue;
            }
            let event = decode_and_validate_lifecycle_row(&row)?;
            event.operation.apply(&mut rebuilt)?;
        }
        Ok(rebuilt)
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
            stored_at_unix_ms: next_ledger_transaction_time_unix_ms(&transaction)?,
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
        Self::search_index(
            &self.index,
            request,
            current_unix_ms()?,
            &self.payload_root,
        )
    }
}

impl SqliteAmlMemoryBackend {
    fn search_index(
        index: &EvidenceIndex,
        request: &AmlSearchRequest,
        valid_at_unix_ms: u64,
        payload_root: &Path,
    ) -> Result<AmlSearchResponse, String> {
        let hits = index.search_bm25(
            &request.query,
            &request.user_id,
            valid_at_unix_ms,
            payload_root,
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
            // When a source message supplies a world-time timestamp, it is
            // also the earliest valid time for that evidence. Transaction-time
            // visibility remains independently governed by the Add commit time.
            valid_from_unix_ms: message_timestamp,
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

fn decode_and_validate_lifecycle_row(
    row: &PersistedLifecycleRow,
) -> Result<PersistedEvidenceLifecycleEvent, String> {
    if row.sequence <= 0 || row.recorded_at_unix_ms < 0 {
        return Err("SQLite lifecycle row has an invalid sequence or timestamp".into());
    }
    let event: PersistedEvidenceLifecycleEvent = serde_json::from_slice(&row.body_json)
        .map_err(|error| format!("persisted AML lifecycle event is invalid JSON: {error}"))?;
    require_non_empty(&event.event_id, "lifecycle event_id")?;
    event.operation.validate()?;
    if sha256_hex(&row.body_json) != row.event_sha256 {
        return Err("SQLite lifecycle event content digest mismatch".into());
    }
    if event.event_id != row.event_id
        || u64::try_from(row.recorded_at_unix_ms).ok() != Some(event.recorded_at_unix_ms)
    {
        return Err("SQLite lifecycle row metadata does not match canonical event".into());
    }
    Ok(event)
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

/// Allocate a transaction time that is monotonic across Add and lifecycle
/// commits, even when wall-clock time repeats or moves backwards. BEGIN
/// IMMEDIATE serializes writers before this global ledger coordinate is read.
fn next_ledger_transaction_time_unix_ms(
    transaction: &rusqlite::Transaction<'_>,
) -> Result<u64, String> {
    let latest_add: Option<i64> = transaction
        .query_row("SELECT MAX(stored_at_unix_ms) FROM aml_add_batches", [], |row| row.get(0))
        .map_err(|error| format!("failed to read latest Add transaction time: {error}"))?;
    let latest_lifecycle: Option<i64> = transaction
        .query_row("SELECT MAX(recorded_at_unix_ms) FROM aml_evidence_lifecycle_events", [], |row| row.get(0))
        .map_err(|error| format!("failed to read latest lifecycle transaction time: {error}"))?;
    let latest = latest_add.into_iter()
        .chain(latest_lifecycle)
        .max();
    let next_logical = match latest {
        Some(last) => u64::try_from(last)
            .map_err(|_| "ledger contains a negative transaction timestamp".to_string())?
            .checked_add(1)
            .ok_or_else(|| "ledger transaction timestamp overflow".to_string())?,
        None => 0,
    };
    Ok(current_unix_ms()?.max(next_logical))
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
    fn database_uses_wal_and_full_synchronous_commit_mode() {
        let root = tempdir().unwrap();
        let backend = open_backend(root.path());
        let journal_mode: String = backend.connection
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        let synchronous: i64 = backend.connection
            .pragma_query_value(None, "synchronous", |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
        assert_eq!(synchronous, 2, "SQLite FULL synchronous mode is numeric value 2");
    }

    #[test]
    fn second_connection_replays_same_add_without_duplicate_batch() {
        let root = tempdir().unwrap();
        let mut first = open_backend(root.path());
        let mut second = open_backend(root.path());
        let request = add_request(
            "request-shared",
            "user-1",
            "session-1",
            &["same logical trajectory committed once"],
        );
        execute_aml_add(&mut first, request.clone()).unwrap();
        execute_aml_add(&mut second, request.clone()).unwrap();
        assert_eq!(first.persisted_batch_count().unwrap(), 1);
        assert_eq!(second.persisted_batch_count().unwrap(), 1);
        let hits = execute_aml_search(
            &second,
            AmlSearchRequest {
                query: "same logical trajectory".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
        ).unwrap();
        assert_eq!(hits.data.len(), 1);
    }

    #[test]
    fn lexical_invalid_message_does_not_partially_commit_add_batch() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let request = add_request(
            "request-invalid",
            "user-1",
            "session-1",
            &["valid evidence message", "!!!"],
        );
        assert!(execute_aml_add(&mut backend, request).is_err());
        assert_eq!(backend.persisted_batch_count().unwrap(), 0);
        let hits = execute_aml_search(
            &backend,
            AmlSearchRequest {
                query: "valid evidence message".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
        ).unwrap();
        assert!(hits.data.is_empty());
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
    fn revoke_lifecycle_event_survives_restart_and_suppresses_search() {
        let root = tempdir().unwrap();
        {
            let mut backend = open_backend(root.path());
            execute_aml_add(
                &mut backend,
                add_request("request-1", "user-1", "session-1", &["secret alpha legacy configuration"]),
            ).unwrap();
            let evidence_id = backend.lookup_existing("request-1").unwrap().unwrap()
                .records[0].evidence_id.clone();
            backend.revoke_evidence("revoke-event-1", &evidence_id).unwrap();
            backend.revoke_evidence("revoke-event-1", &evidence_id).unwrap();
            assert_eq!(backend.lifecycle_event_count().unwrap(), 1);
        }
        let backend = open_backend(root.path());
        assert_eq!(backend.lifecycle_event_count().unwrap(), 1);
        let result = execute_aml_search(&backend, AmlSearchRequest {
            query: "alpha legacy configuration".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        assert!(result.data.is_empty());
    }

    #[test]
    fn lifecycle_event_id_cannot_be_reused_for_a_different_operation() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request("request-1", "user-1", "session-1", &["legacy cache token"]),
        ).unwrap();
        let evidence_id = backend.lookup_existing("request-1").unwrap().unwrap()
            .records[0].evidence_id.clone();
        backend.revoke_evidence("event-1", &evidence_id).unwrap();
        assert!(backend.delete_evidence("event-1", &evidence_id).is_err());
        assert_eq!(backend.lifecycle_event_count().unwrap(), 1);
    }

    #[test]
    fn supersession_lifecycle_event_survives_restart() {
        let root = tempdir().unwrap();
        {
            let mut backend = open_backend(root.path());
            execute_aml_add(
                &mut backend,
                add_request("old-request", "user-1", "session-1", &["alpha legacy resolver configuration"]),
            ).unwrap();
            execute_aml_add(
                &mut backend,
                add_request("new-request", "user-1", "session-2", &["omega replacement migration policy"]),
            ).unwrap();
            let old_id = backend.lookup_existing("old-request").unwrap().unwrap()
                .records[0].evidence_id.clone();
            let new_id = backend.lookup_existing("new-request").unwrap().unwrap()
                .records[0].evidence_id.clone();
            backend.supersede_evidence("supersede-event-1", &old_id, &new_id).unwrap();
            assert_eq!(backend.lifecycle_event_count().unwrap(), 1);
        }
        let backend = open_backend(root.path());
        let old = execute_aml_search(&backend, AmlSearchRequest {
            query: "alpha legacy resolver".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        let new = execute_aml_search(&backend, AmlSearchRequest {
            query: "omega replacement migration".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        }).unwrap();
        assert!(old.data.is_empty());
        assert_eq!(new.data.len(), 1);
    }

    #[test]
    fn retention_purge_removes_expired_sources_and_dependent_lifecycle_events() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request("request-1", "user-1", "session-1", &["old retention test payload"]),
        ).unwrap();
        let evidence_id = backend.lookup_existing("request-1").unwrap().unwrap()
            .records[0].evidence_id.clone();
        backend.revoke_evidence("revoke-event-1", &evidence_id).unwrap();
        assert_eq!(backend.lifecycle_event_count().unwrap(), 1);
        let cutoff = current_unix_ms().unwrap().saturating_add(1);
        assert_eq!(backend.purge_stored_before(cutoff).unwrap(), 1);
        assert_eq!(backend.persisted_batch_count().unwrap(), 0);
        assert_eq!(backend.lifecycle_event_count().unwrap(), 0);
    }

    #[test]
    fn add_and_lifecycle_transaction_times_are_strictly_monotonic() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        execute_aml_add(
            &mut backend,
            add_request("request-1", "user-1", "session-1", &["first immutable memory event"]),
        ).unwrap();
        let first = backend.lookup_existing("request-1").unwrap().unwrap()
            .stored_at_unix_ms;

        execute_aml_add(
            &mut backend,
            add_request("request-2", "user-1", "session-2", &["second immutable memory event"]),
        ).unwrap();
        let second = backend.lookup_existing("request-2").unwrap().unwrap()
            .stored_at_unix_ms;
        assert!(second > first, "Add transaction time must be strictly increasing");

        let first_id = backend.lookup_existing("request-1").unwrap().unwrap()
            .records[0].evidence_id.clone();
        backend.revoke_evidence("revoke-monotonic", &first_id).unwrap();
        let third: i64 = backend.connection.query_row(
            "SELECT recorded_at_unix_ms FROM aml_evidence_lifecycle_events WHERE event_id = ?1",
            params!["revoke-monotonic"],
            |row| row.get(0),
        ).unwrap();
        assert!(u64::try_from(third).unwrap() > second);
    }

    #[test]
    fn schema_v1_add_store_migrates_to_v2_lifecycle_table() {
        let root = tempdir().unwrap();
        let path = root.path().join("v1.sqlite3");
        {
            let connection = Connection::open(&path).unwrap();
            connection.execute_batch(
                "CREATE TABLE aml_add_batches (
                    request_id TEXT PRIMARY KEY NOT NULL COLLATE BINARY,
                    payload_sha256 TEXT NOT NULL,
                    stored_at_unix_ms INTEGER NOT NULL,
                    body_json BLOB NOT NULL
                );
                PRAGMA user_version = 1;"
            ).unwrap();
        }
        let backend = SqliteAmlMemoryBackend::open(&path, root.path()).unwrap();
        let version: i64 = backend.connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(version, SQLITE_SCHEMA_VERSION);
        assert_eq!(backend.lifecycle_event_count().unwrap(), 0);
    }

    #[test]
    fn late_arriving_memory_is_hidden_before_ingestion_and_visible_after() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let mut request = add_request(
            "late-source",
            "user-1",
            "session-1",
            &["A historical schema migration fixed the parser."],
        );
        request.messages[0].timestamp = Some(1_700_000_000_000);
        execute_aml_add(&mut backend, request).unwrap();
        let batch = backend.lookup_existing("late-source").unwrap().unwrap();
        let stored_at = batch.stored_at_unix_ms;
        let valid_at = 1_750_000_000_000;

        let before_ingestion = backend.search_as_known_at(
            &AmlSearchRequest {
                query: "historical schema migration parser".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
            valid_at,
            stored_at.saturating_sub(1),
        ).unwrap();
        assert!(before_ingestion.data.is_empty());

        let after_ingestion = backend.search_as_known_at(
            &AmlSearchRequest {
                query: "historical schema migration parser".into(),
                options: None,
                user_id: "user-1".into(),
                top_k: 10,
            },
            valid_at,
            stored_at,
        ).unwrap();
        assert_eq!(after_ingestion.data.len(), 1);
    }

    #[test]
    fn retraction_changes_as_known_view_only_after_recorded_time() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let mut request = add_request(
            "revoked-source",
            "user-1",
            "session-1",
            &["legacy deployment procedure uses unsafe token"],
        );
        request.messages[0].timestamp = Some(1_700_000_000_000);
        execute_aml_add(&mut backend, request).unwrap();
        let evidence_id = backend.lookup_existing("revoked-source").unwrap().unwrap()
            .records[0].evidence_id.clone();
        std::thread::sleep(Duration::from_millis(3));
        backend.revoke_evidence("revoke-at-t2", &evidence_id).unwrap();
        let retracted_at: i64 = backend.connection.query_row(
            "SELECT recorded_at_unix_ms FROM aml_evidence_lifecycle_events WHERE event_id = ?1",
            params!["revoke-at-t2"],
            |row| row.get(0),
        ).unwrap();
        let retracted_at = u64::try_from(retracted_at).unwrap();

        let query = AmlSearchRequest {
            query: "legacy deployment unsafe token".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        };
        let before_retraction_was_known = backend.search_as_known_at(
            &query,
            1_780_000_000_000,
            retracted_at.saturating_sub(1),
        ).unwrap();
        assert_eq!(before_retraction_was_known.data.len(), 1);

        let after_retraction_was_known = backend.search_as_known_at(
            &query,
            1_780_000_000_000,
            retracted_at,
        ).unwrap();
        assert!(after_retraction_was_known.data.is_empty());
    }

    #[test]
    fn world_valid_time_and_system_known_time_are_independent() {
        let root = tempdir().unwrap();
        let mut backend = open_backend(root.path());
        let valid_from = 1_700_000_000_000u64;
        let mut request = add_request(
            "future-valid-source",
            "user-1",
            "session-1",
            &["New policy becomes valid on the announced date."],
        );
        request.messages[0].timestamp = Some(valid_from as i64);
        execute_aml_add(&mut backend, request).unwrap();
        let stored_at = backend.lookup_existing("future-valid-source").unwrap().unwrap()
            .stored_at_unix_ms;
        let query = AmlSearchRequest {
            query: "policy announced date".into(),
            options: None,
            user_id: "user-1".into(),
            top_k: 10,
        };

        let before_world_validity = backend.search_as_known_at(
            &query,
            valid_from - 1,
            stored_at,
        ).unwrap();
        assert!(before_world_validity.data.is_empty());

        let once_world_valid = backend.search_as_known_at(
            &query,
            valid_from,
            stored_at,
        ).unwrap();
        assert_eq!(once_world_valid.data.len(), 1);
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
