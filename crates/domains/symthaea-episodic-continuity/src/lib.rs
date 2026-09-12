// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact SQLite persistence for canonical episodic restart continuity.
//!
//! This crate is deliberately separate from Symthaea's similarity/search memory database. The
//! search projection is allowed to be lossy; restart continuity is not. Here, exact occurrence
//! UUIDs, versioned episode envelopes, reversible quarantine escrow, and both quarantine ledgers
//! are stored without reinterpreting their semantics.
//!
//! SQLite durability and hash chaining are not called rollback resistance. Recovery still requires
//! externally retained trusted ledger heads (TPM/TEE, Xenia/Mycelix, or another monotonic anchor).

#![deny(unsafe_code)]

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_welfare_assurance::memory_quarantine::{
    EpisodicQuarantineEscrow, EpisodicQuarantineEscrowPersistence,
    digest_episodic_quarantine_escrow,
};
use symthaea_welfare_assurance::persisted_episode_envelope::{
    CanonicalEpisodicImportBatch, PersistedEpisodicEnvelope, ValidatedPersistedOccurrence,
    materialize_canonical_import_batch,
};
use symthaea_welfare_assurance::quarantine_intent_ledger::{
    EpisodicQuarantineIntentLedger, QuarantineIntentEnvelope,
};
use symthaea_welfare_assurance::quarantine_intent_persistence::QuarantineIntentLedgerPersistence;
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerEnvelope,
};
use symthaea_welfare_assurance::restart_activation_gate::{
    EpisodicRestartActivationPlan, PersistedQuarantineEscrowDescriptor,
    build_episodic_restart_activation_plan,
};
use thiserror::Error;

pub const CONTINUITY_DB_SCHEMA: &str = "symthaea.episodic-continuity.sqlite.v1";
const MAX_LEDGER_BYTES: usize = 32 * 1024 * 1024;
const MAX_ESCROW_BYTES: usize = 32 * 1024 * 1024;

/// Exact continuity store. This database is a constitutional/restart substrate, not a search index.
pub struct SqliteEpisodicContinuityStore {
    conn: Connection,
}

impl SqliteEpisodicContinuityStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContinuityStoreError> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| ContinuityStoreError::Io(error.to_string()))?;
            }
        }
        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    pub fn in_memory() -> Result<Self, ContinuityStoreError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self, ContinuityStoreError> {
        conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = FULL;

            CREATE TABLE IF NOT EXISTS continuity_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodic_occurrences (
                record_id TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL UNIQUE,
                envelope BLOB NOT NULL,
                updated_at_unix_s INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quarantine_escrows (
                instance_id TEXT PRIMARY KEY,
                escrow BLOB NOT NULL,
                escrow_digest BLOB NOT NULL,
                persistence_ref TEXT NOT NULL,
                updated_at_unix_s INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quarantine_intent_ledger (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                events BLOB NOT NULL,
                head_hash BLOB NOT NULL,
                generation INTEGER NOT NULL,
                persistence_ref TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quarantine_state_ledger (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                events BLOB NOT NULL,
                head_hash BLOB NOT NULL,
                generation INTEGER NOT NULL,
                persistence_ref TEXT NOT NULL
            );
            "#,
        )?;

        conn.execute(
            "INSERT OR IGNORE INTO continuity_meta(key, value) VALUES('schema', ?1)",
            [CONTINUITY_DB_SCHEMA],
        )?;
        let schema: String = conn.query_row(
            "SELECT value FROM continuity_meta WHERE key = 'schema'",
            [],
            |row| row.get(0),
        )?;
        if schema != CONTINUITY_DB_SCHEMA {
            return Err(ContinuityStoreError::UnsupportedDatabaseSchema(schema));
        }
        Ok(Self { conn })
    }

    /// Upsert one exact active-record envelope by stable occurrence UUID.
    ///
    /// Repeated flushes of the same occurrence replace one row; rank changes can no longer create
    /// new continuity identities.
    pub fn upsert_occurrence(
        &mut self,
        envelope: &PersistedEpisodicEnvelope,
    ) -> Result<String, ContinuityStoreError> {
        envelope.validate()?;
        let encoded = envelope.encode()?;
        let record_id = envelope.stable_record_id();
        let instance_id = envelope.instance_id.to_string();
        self.conn.execute(
            r#"
            INSERT INTO episodic_occurrences(record_id, instance_id, envelope, updated_at_unix_s)
            VALUES(?1, ?2, ?3, ?4)
            ON CONFLICT(record_id) DO UPDATE SET
                instance_id = excluded.instance_id,
                envelope = excluded.envelope,
                updated_at_unix_s = excluded.updated_at_unix_s
            "#,
            params![
                record_id,
                instance_id,
                encoded,
                u64_to_i64(envelope.persisted_at_unix_s, "persisted_at_unix_s")?
            ],
        )?;
        Ok(occurrence_ref(&record_id))
    }

    /// Load and validate every exact persisted occurrence in deterministic primary-key order.
    pub fn load_validated_occurrences(
        &self,
    ) -> Result<Vec<ValidatedPersistedOccurrence>, ContinuityStoreError> {
        let mut statement = self.conn.prepare(
            "SELECT record_id, instance_id, envelope FROM episodic_occurrences ORDER BY record_id",
        )?;
        let mut rows = statement.query([])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let record_id: String = row.get(0)?;
            let stored_instance_id: String = row.get(1)?;
            let encoded: Vec<u8> = row.get(2)?;
            let envelope = PersistedEpisodicEnvelope::decode(&encoded)?;
            if envelope.stable_record_id() != record_id
                || envelope.instance_id.to_string() != stored_instance_id
            {
                return Err(ContinuityStoreError::OccurrenceRowIdentityMismatch {
                    record_id,
                });
            }
            result.push(envelope.validate_record_ref(occurrence_ref(&record_id))?);
        }
        Ok(result)
    }

    /// Recover anchored quarantine intent state. Empty storage is valid only when the externally
    /// trusted head is the ledger's genesis head.
    pub fn recover_intent_ledger(
        &self,
        expected_trusted_head: Sha256Digest,
    ) -> Result<EpisodicQuarantineIntentLedger, ContinuityStoreError> {
        let stored = self
            .conn
            .query_row(
                "SELECT events, head_hash FROM quarantine_intent_ledger WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        let Some((events_bytes, stored_head_bytes)) = stored else {
            return EpisodicQuarantineIntentLedger::recover_anchored(&[], expected_trusted_head)
                .map_err(|error| ContinuityStoreError::IntentLedger(error.to_string()));
        };
        if events_bytes.len() > MAX_LEDGER_BYTES {
            return Err(ContinuityStoreError::StoredBlobTooLarge {
                field: "quarantine_intent_ledger.events",
                actual: events_bytes.len(),
                max: MAX_LEDGER_BYTES,
            });
        }
        let events: Vec<QuarantineIntentEnvelope> = bincode::deserialize(&events_bytes)
            .map_err(|error| ContinuityStoreError::Decoding(error.to_string()))?;
        let stored_head = decode_digest(&stored_head_bytes, "quarantine_intent_ledger.head_hash")?;
        let recovered = EpisodicQuarantineIntentLedger::recover_anchored(
            &events,
            expected_trusted_head,
        )
        .map_err(|error| ContinuityStoreError::IntentLedger(error.to_string()))?;
        if recovered.head_hash() != stored_head {
            return Err(ContinuityStoreError::StoredHeadMismatch {
                field: "quarantine_intent_ledger.head_hash",
            });
        }
        Ok(recovered)
    }

    pub fn recover_quarantine_ledger(
        &self,
        expected_trusted_head: Sha256Digest,
    ) -> Result<EpisodicQuarantineStateLedger, ContinuityStoreError> {
        let stored = self
            .conn
            .query_row(
                "SELECT events, head_hash FROM quarantine_state_ledger WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        let Some((events_bytes, stored_head_bytes)) = stored else {
            return EpisodicQuarantineStateLedger::recover_anchored(&[], expected_trusted_head)
                .map_err(|error| ContinuityStoreError::QuarantineLedger(error.to_string()));
        };
        if events_bytes.len() > MAX_LEDGER_BYTES {
            return Err(ContinuityStoreError::StoredBlobTooLarge {
                field: "quarantine_state_ledger.events",
                actual: events_bytes.len(),
                max: MAX_LEDGER_BYTES,
            });
        }
        let events: Vec<QuarantineLedgerEnvelope> = bincode::deserialize(&events_bytes)
            .map_err(|error| ContinuityStoreError::Decoding(error.to_string()))?;
        let stored_head = decode_digest(&stored_head_bytes, "quarantine_state_ledger.head_hash")?;
        let recovered = EpisodicQuarantineStateLedger::recover_anchored(
            &events,
            expected_trusted_head,
        )
        .map_err(|error| ContinuityStoreError::QuarantineLedger(error.to_string()))?;
        if recovered.head_hash() != stored_head {
            return Err(ContinuityStoreError::StoredHeadMismatch {
                field: "quarantine_state_ledger.head_hash",
            });
        }
        Ok(recovered)
    }

    pub fn load_escrow_descriptors(
        &self,
    ) -> Result<Vec<PersistedQuarantineEscrowDescriptor>, ContinuityStoreError> {
        let mut statement = self.conn.prepare(
            "SELECT instance_id, escrow, escrow_digest, persistence_ref FROM quarantine_escrows ORDER BY instance_id",
        )?;
        let mut rows = statement.query([])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let stored_instance_id: String = row.get(0)?;
            let encoded: Vec<u8> = row.get(1)?;
            let digest_bytes: Vec<u8> = row.get(2)?;
            let persistence_ref: String = row.get(3)?;
            if encoded.len() > MAX_ESCROW_BYTES {
                return Err(ContinuityStoreError::StoredBlobTooLarge {
                    field: "quarantine_escrows.escrow",
                    actual: encoded.len(),
                    max: MAX_ESCROW_BYTES,
                });
            }
            let escrow: EpisodicQuarantineEscrow = bincode::deserialize(&encoded)
                .map_err(|error| ContinuityStoreError::Decoding(error.to_string()))?;
            escrow.validate()?;
            if escrow.instance_id.to_string() != stored_instance_id {
                return Err(ContinuityStoreError::EscrowRowIdentityMismatch {
                    instance_id: stored_instance_id,
                });
            }
            let stored_digest = decode_digest(&digest_bytes, "quarantine_escrows.escrow_digest")?;
            let actual_digest = digest_episodic_quarantine_escrow(&escrow)?;
            if stored_digest != actual_digest {
                return Err(ContinuityStoreError::EscrowDigestMismatch(
                    escrow.instance_id.to_string(),
                ));
            }
            result.push(PersistedQuarantineEscrowDescriptor::new(
                escrow.instance_id,
                escrow.content_id,
                actual_digest,
                persistence_ref,
            )?);
        }
        Ok(result)
    }

    /// Recover all exact continuity material and derive the fail-closed restart plan/import batch.
    pub fn recover_restart(
        &self,
        store_target_id: &str,
        expected_intent_head: Sha256Digest,
        expected_quarantine_head: Sha256Digest,
    ) -> Result<RecoveredEpisodicContinuity, ContinuityStoreError> {
        let occurrences = self.load_validated_occurrences()?;
        let escrows = self.load_escrow_descriptors()?;
        let intent_ledger = self.recover_intent_ledger(expected_intent_head)?;
        let quarantine_ledger = self.recover_quarantine_ledger(expected_quarantine_head)?;
        let descriptors: Vec<_> = occurrences.iter().map(|record| record.descriptor()).collect();
        let activation_plan = build_episodic_restart_activation_plan(
            store_target_id,
            &descriptors,
            &escrows,
            &intent_ledger,
            &quarantine_ledger,
        )?;
        let import_batch =
            materialize_canonical_import_batch(store_target_id, &activation_plan, &occurrences)?;
        Ok(RecoveredEpisodicContinuity {
            intent_ledger,
            quarantine_ledger,
            activation_plan,
            import_batch,
        })
    }
}

impl EpisodicQuarantineEscrowPersistence for SqliteEpisodicContinuityStore {
    type Error = ContinuityStoreError;

    fn persist_episodic_quarantine_escrow(
        &mut self,
        escrow: &EpisodicQuarantineEscrow,
        escrow_digest: Sha256Digest,
    ) -> Result<String, Self::Error> {
        escrow.validate()?;
        let actual_digest = digest_episodic_quarantine_escrow(escrow)?;
        if actual_digest != escrow_digest {
            return Err(ContinuityStoreError::SuppliedEscrowDigestMismatch);
        }
        let encoded = bincode::serialize(escrow)
            .map_err(|error| ContinuityStoreError::Encoding(error.to_string()))?;
        if encoded.len() > MAX_ESCROW_BYTES {
            return Err(ContinuityStoreError::StoredBlobTooLarge {
                field: "quarantine_escrows.escrow",
                actual: encoded.len(),
                max: MAX_ESCROW_BYTES,
            });
        }
        let persistence_ref = format!(
            "sqlite-continuity:escrow:{}:sha256:{}",
            escrow.instance_id,
            hex_digest(escrow_digest)
        );
        self.conn.execute(
            r#"
            INSERT INTO quarantine_escrows(instance_id, escrow, escrow_digest, persistence_ref, updated_at_unix_s)
            VALUES(?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(instance_id) DO UPDATE SET
                escrow = excluded.escrow,
                escrow_digest = excluded.escrow_digest,
                persistence_ref = excluded.persistence_ref,
                updated_at_unix_s = excluded.updated_at_unix_s
            "#,
            params![
                escrow.instance_id.to_string(),
                encoded,
                escrow_digest.0.to_vec(),
                persistence_ref,
                u64_to_i64(escrow.captured_at_unix_s, "captured_at_unix_s")?
            ],
        )?;
        Ok(persistence_ref)
    }
}

impl QuarantineIntentLedgerPersistence for SqliteEpisodicContinuityStore {
    type Error = ContinuityStoreError;

    fn persist_quarantine_intent_ledger(
        &mut self,
        events: &[QuarantineIntentEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        let recovered = EpisodicQuarantineIntentLedger::recover_anchored(events, head_hash)
            .map_err(|error| ContinuityStoreError::IntentLedger(error.to_string()))?;
        let encoded = bincode::serialize(events)
            .map_err(|error| ContinuityStoreError::Encoding(error.to_string()))?;
        if encoded.len() > MAX_LEDGER_BYTES {
            return Err(ContinuityStoreError::StoredBlobTooLarge {
                field: "quarantine_intent_ledger.events",
                actual: encoded.len(),
                max: MAX_LEDGER_BYTES,
            });
        }
        let persistence_ref = format!(
            "sqlite-continuity:intent-ledger:{}:sha256:{}",
            recovered.generation(),
            hex_digest(head_hash)
        );
        self.conn.execute(
            r#"
            INSERT INTO quarantine_intent_ledger(singleton, events, head_hash, generation, persistence_ref)
            VALUES(1, ?1, ?2, ?3, ?4)
            ON CONFLICT(singleton) DO UPDATE SET
                events = excluded.events,
                head_hash = excluded.head_hash,
                generation = excluded.generation,
                persistence_ref = excluded.persistence_ref
            "#,
            params![
                encoded,
                head_hash.0.to_vec(),
                u64_to_i64(recovered.generation(), "intent_generation")?,
                persistence_ref
            ],
        )?;
        Ok(persistence_ref)
    }
}

impl QuarantineLedgerPersistence for SqliteEpisodicContinuityStore {
    type Error = ContinuityStoreError;

    fn persist_quarantine_ledger(
        &mut self,
        events: &[QuarantineLedgerEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        let recovered = EpisodicQuarantineStateLedger::recover_anchored(events, head_hash)
            .map_err(|error| ContinuityStoreError::QuarantineLedger(error.to_string()))?;
        let encoded = bincode::serialize(events)
            .map_err(|error| ContinuityStoreError::Encoding(error.to_string()))?;
        if encoded.len() > MAX_LEDGER_BYTES {
            return Err(ContinuityStoreError::StoredBlobTooLarge {
                field: "quarantine_state_ledger.events",
                actual: encoded.len(),
                max: MAX_LEDGER_BYTES,
            });
        }
        let persistence_ref = format!(
            "sqlite-continuity:state-ledger:{}:sha256:{}",
            recovered.generation(),
            hex_digest(head_hash)
        );
        self.conn.execute(
            r#"
            INSERT INTO quarantine_state_ledger(singleton, events, head_hash, generation, persistence_ref)
            VALUES(1, ?1, ?2, ?3, ?4)
            ON CONFLICT(singleton) DO UPDATE SET
                events = excluded.events,
                head_hash = excluded.head_hash,
                generation = excluded.generation,
                persistence_ref = excluded.persistence_ref
            "#,
            params![
                encoded,
                head_hash.0.to_vec(),
                u64_to_i64(recovered.generation(), "state_generation")?,
                persistence_ref
            ],
        )?;
        Ok(persistence_ref)
    }
}

/// Fully validated restart material. This still does not mutate the private replay heap.
pub struct RecoveredEpisodicContinuity {
    pub intent_ledger: EpisodicQuarantineIntentLedger,
    pub quarantine_ledger: EpisodicQuarantineStateLedger,
    pub activation_plan: EpisodicRestartActivationPlan,
    pub import_batch: CanonicalEpisodicImportBatch,
}

fn occurrence_ref(record_id: &str) -> String {
    format!("sqlite-continuity:occurrence:{record_id}")
}

fn u64_to_i64(value: u64, field: &'static str) -> Result<i64, ContinuityStoreError> {
    i64::try_from(value).map_err(|_| ContinuityStoreError::IntegerOutOfRange { field, value })
}

fn decode_digest(bytes: &[u8], field: &'static str) -> Result<Sha256Digest, ContinuityStoreError> {
    if bytes.len() != 32 {
        return Err(ContinuityStoreError::InvalidDigestLength {
            field,
            actual: bytes.len(),
        });
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(bytes);
    Ok(Sha256Digest(digest))
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[derive(Debug, Error)]
pub enum ContinuityStoreError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("continuity database IO failed: {0}")]
    Io(String),
    #[error("unsupported continuity database schema: {0:?}")]
    UnsupportedDatabaseSchema(String),
    #[error("continuity encoding failed: {0}")]
    Encoding(String),
    #[error("continuity decoding failed: {0}")]
    Decoding(String),
    #[error("stored continuity blob `{field}` is too large: actual={actual}, max={max}")]
    StoredBlobTooLarge {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    #[error("continuity integer `{field}` is outside SQLite i64 range: {value}")]
    IntegerOutOfRange { field: &'static str, value: u64 },
    #[error("stored digest `{field}` has invalid length: {actual}")]
    InvalidDigestLength { field: &'static str, actual: usize },
    #[error("persisted occurrence row identity does not match envelope: {record_id:?}")]
    OccurrenceRowIdentityMismatch { record_id: String },
    #[error("persisted escrow row identity mismatch: {instance_id:?}")]
    EscrowRowIdentityMismatch { instance_id: String },
    #[error("persisted escrow digest mismatch for occurrence {0}")]
    EscrowDigestMismatch(String),
    #[error("caller supplied escrow digest does not match escrow bytes")]
    SuppliedEscrowDigestMismatch,
    #[error("stored ledger head disagrees with recovered event chain for `{field}`")]
    StoredHeadMismatch { field: &'static str },
    #[error("quarantine intent ledger validation failed: {0}")]
    IntentLedger(String),
    #[error("quarantine state ledger validation failed: {0}")]
    QuarantineLedger(String),
    #[error(transparent)]
    PersistedEnvelope(#[from] symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodeEnvelopeError),
    #[error(transparent)]
    RestartActivation(#[from] symthaea_welfare_assurance::restart_activation_gate::RestartActivationError),
    #[error(transparent)]
    ImportBatch(#[from] symthaea_welfare_assurance::persisted_episode_envelope::CanonicalImportBatchError),
    #[error(transparent)]
    QuarantineIntervention(#[from] symthaea_welfare_assurance::memory_quarantine::EpisodicQuarantineInterventionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
    use symthaea_welfare_assurance::memory_identity::episode_content_id;
    use symthaea_welfare_assurance::memory_quarantine::{
        EPISODIC_QUARANTINE_ESCROW_SCHEMA, episodic_instance_target_id,
    };

    const STORE: &str = "symthaea:self:episodic-memory";

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn duplicate_pair() -> (Episode, Episode) {
        let source = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
            ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
            0.85,
            42,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(source.clone()).unwrap();
        memory.store_if_significant_with_id(source).unwrap();
        let mut episodes: Vec<_> = memory
            .get_top_episode_instances(10)
            .into_iter()
            .map(|(_, episode)| episode)
            .collect();
        episodes.sort_by_key(|episode| episode.instance_id.unwrap());
        (episodes.remove(0), episodes.remove(0))
    }

    #[test]
    fn stable_occurrence_rows_survive_repeated_upsert_without_rank_identity() {
        let (mut first, second) = duplicate_pair();
        let first_id = first.instance_id.unwrap();
        let second_id = second.instance_id.unwrap();
        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        let envelope = PersistedEpisodicEnvelope::new(STORE, first.clone(), 100, 1).unwrap();
        let first_ref = store.upsert_occurrence(&envelope).unwrap();
        first.replay_count = 3;
        let updated = PersistedEpisodicEnvelope::new(STORE, first, 120, 2).unwrap();
        let second_ref = store.upsert_occurrence(&updated).unwrap();
        store
            .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, second, 120, 2).unwrap())
            .unwrap();

        assert_eq!(first_ref, second_ref);
        let loaded = store.load_validated_occurrences().unwrap();
        assert_eq!(loaded.len(), 2);
        let first_loaded = loaded.iter().find(|row| row.instance_id() == first_id).unwrap();
        assert_eq!(first_loaded.instance_id(), first_id);
        assert_eq!(first_loaded.episode().replay_count, 3);
        assert!(loaded.iter().any(|row| row.instance_id() == second_id));
    }

    #[test]
    fn exact_duplicate_quarantine_recovers_one_inactive_and_one_active() {
        let (first, second) = duplicate_pair();
        let first_id = first.instance_id.unwrap();
        let second_id = second.instance_id.unwrap();
        let content_id = episode_content_id(&first).unwrap();
        assert_eq!(content_id, episode_content_id(&second).unwrap());
        let target = episodic_instance_target_id(STORE, first_id).unwrap();

        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        store
            .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, first.clone(), 100, 1).unwrap())
            .unwrap();
        store
            .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, second, 100, 1).unwrap())
            .unwrap();

        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: first_id,
            content_id,
            captured_at_unix_s: 110,
            pre_active_state_digest: digest(8),
            episode: first,
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let escrow_ref = store
            .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
            .unwrap();

        let mut state = EpisodicQuarantineStateLedger::new();
        state
            .append_quarantined(
                &target,
                first_id,
                content_id,
                111,
                escrow_digest,
                &escrow_ref,
            )
            .unwrap();
        store
            .persist_quarantine_ledger(state.events(), state.head_hash())
            .unwrap();

        let mut intent = EpisodicQuarantineIntentLedger::new();
        intent
            .append_prepared(
                "exec:q:1",
                &target,
                first_id,
                content_id,
                110,
                digest(7),
                escrow_digest,
                &escrow_ref,
            )
            .unwrap();
        intent
            .append_committed(
                "exec:q:1",
                &target,
                first_id,
                content_id,
                112,
                state.head_hash(),
            )
            .unwrap();
        store
            .persist_quarantine_intent_ledger(intent.events(), intent.head_hash())
            .unwrap();

        let recovered = store
            .recover_restart(STORE, intent.head_hash(), state.head_hash())
            .unwrap();
        assert_eq!(recovered.activation_plan.active.len(), 1);
        assert_eq!(recovered.activation_plan.active[0].instance_id, second_id);
        assert_eq!(recovered.activation_plan.inactive.len(), 1);
        assert_eq!(recovered.activation_plan.inactive[0].instance_id, first_id);
        assert_eq!(recovered.import_batch.active().len(), 1);
        assert_eq!(recovered.import_batch.active()[0].instance_id(), second_id);
        assert_eq!(recovered.import_batch.withheld().len(), 1);
        assert_eq!(recovered.import_batch.withheld()[0].instance_id, first_id);
    }

    #[test]
    fn external_anchor_detects_locally_valid_intent_rollback() {
        let (first, _) = duplicate_pair();
        let first_id = first.instance_id.unwrap();
        let content_id = episode_content_id(&first).unwrap();
        let target = episodic_instance_target_id(STORE, first_id).unwrap();
        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();

        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: first_id,
            content_id,
            captured_at_unix_s: 100,
            pre_active_state_digest: digest(2),
            episode: first,
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let escrow_ref = store
            .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
            .unwrap();

        let mut one_event = EpisodicQuarantineIntentLedger::new();
        one_event
            .append_prepared(
                "exec:q:rollback",
                target,
                first_id,
                content_id,
                101,
                digest(3),
                escrow_digest,
                escrow_ref,
            )
            .unwrap();
        let old_head = one_event.head_hash();
        store
            .persist_quarantine_intent_ledger(one_event.events(), old_head)
            .unwrap();

        let mut later = one_event.clone();
        later
            .append_aborted(
                "exec:q:rollback",
                later.pending_state(first_id).unwrap().target_id.clone(),
                first_id,
                content_id,
                102,
                digest(4),
            )
            .unwrap();
        let trusted_later_head = later.head_hash();
        assert_ne!(old_head, trusted_later_head);

        assert!(matches!(
            store.recover_intent_ledger(trusted_later_head),
            Err(ContinuityStoreError::IntentLedger(_))
        ));
    }
}