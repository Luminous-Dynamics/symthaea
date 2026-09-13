// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only exact point lookup for persisted episodic quarantine escrow.
//!
//! This crate opens the #2095 continuity database without write authority and exposes only the
//! purpose-separated `EpisodicQuarantineEscrowLookup` contract. It has no enumeration API: callers
//! must supply one exact `EpisodeInstanceId`.

#![deny(unsafe_code)]

use std::path::Path;

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use symthaea_episodic_continuity::CONTINUITY_DB_SCHEMA;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use symthaea_welfare_assurance::memory_quarantine::{
    EpisodicQuarantineEscrow, digest_episodic_quarantine_escrow,
};
use symthaea_welfare_assurance::persisted_memory_restore::{
    EpisodicQuarantineEscrowLookup, PersistedEpisodicEscrowRow,
};
use thiserror::Error;

const MAX_ESCROW_BYTES: usize = 32 * 1024 * 1024;
const MAX_PERSISTENCE_REF_BYTES: usize = 2048;

/// Read-only view over the continuity database's quarantine-escrow table.
pub struct SqliteEpisodicEscrowLookup {
    conn: Connection,
}

impl SqliteEpisodicEscrowLookup {
    /// Open an existing continuity database without write authority.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteEpisodicEscrowLookupError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)?;
        let schema = conn
            .query_row(
                "SELECT value FROM continuity_meta WHERE key = 'schema'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        match schema {
            Some(schema) if schema == CONTINUITY_DB_SCHEMA => Ok(Self { conn }),
            Some(schema) => Err(SqliteEpisodicEscrowLookupError::UnsupportedDatabaseSchema(
                schema,
            )),
            None => Err(SqliteEpisodicEscrowLookupError::MissingDatabaseSchema),
        }
    }
}

impl EpisodicQuarantineEscrowLookup for SqliteEpisodicEscrowLookup {
    type Error = SqliteEpisodicEscrowLookupError;

    fn load_episodic_quarantine_escrow(
        &self,
        instance_id: EpisodeInstanceId,
    ) -> Result<Option<PersistedEpisodicEscrowRow>, Self::Error> {
        let expected_id = instance_id.to_string();
        let row = self
            .conn
            .query_row(
                r#"
                SELECT instance_id, escrow, escrow_digest, persistence_ref
                FROM quarantine_escrows
                WHERE instance_id = ?1
                LIMIT 1
                "#,
                params![expected_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .optional()?;

        let Some((stored_instance_id, encoded, digest_bytes, persistence_ref)) = row else {
            return Ok(None);
        };
        if stored_instance_id != instance_id.to_string() {
            return Err(SqliteEpisodicEscrowLookupError::RowIdentityMismatch {
                requested: instance_id,
                stored: stored_instance_id,
            });
        }
        if encoded.is_empty() || encoded.len() > MAX_ESCROW_BYTES {
            return Err(SqliteEpisodicEscrowLookupError::InvalidEscrowSize {
                actual: encoded.len(),
                max: MAX_ESCROW_BYTES,
            });
        }
        validate_ref(&persistence_ref)?;

        let escrow: EpisodicQuarantineEscrow = bincode::deserialize(&encoded)
            .map_err(|error| SqliteEpisodicEscrowLookupError::Decoding(error.to_string()))?;
        escrow
            .validate()
            .map_err(|error| SqliteEpisodicEscrowLookupError::EscrowValidation(error.to_string()))?;
        if escrow.instance_id != instance_id {
            return Err(SqliteEpisodicEscrowLookupError::PayloadIdentityMismatch {
                requested: instance_id,
                payload: escrow.instance_id,
            });
        }

        let stored_digest = decode_digest(&digest_bytes)?;
        let actual_digest = digest_episodic_quarantine_escrow(&escrow)
            .map_err(|error| SqliteEpisodicEscrowLookupError::EscrowValidation(error.to_string()))?;
        if actual_digest != stored_digest {
            return Err(SqliteEpisodicEscrowLookupError::DigestMismatch);
        }

        Ok(Some(PersistedEpisodicEscrowRow {
            escrow,
            stored_digest,
            persistence_ref,
        }))
    }
}

fn decode_digest(bytes: &[u8]) -> Result<Sha256Digest, SqliteEpisodicEscrowLookupError> {
    if bytes.len() != 32 {
        return Err(SqliteEpisodicEscrowLookupError::InvalidDigestLength {
            actual: bytes.len(),
        });
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(bytes);
    if digest == [0; 32] {
        return Err(SqliteEpisodicEscrowLookupError::ZeroDigest);
    }
    Ok(Sha256Digest(digest))
}

fn validate_ref(value: &str) -> Result<(), SqliteEpisodicEscrowLookupError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_PERSISTENCE_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(SqliteEpisodicEscrowLookupError::InvalidPersistenceReference)
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SqliteEpisodicEscrowLookupError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("continuity database is missing its schema marker")]
    MissingDatabaseSchema,
    #[error("unsupported continuity database schema: {0:?}")]
    UnsupportedDatabaseSchema(String),
    #[error("escrow row identity mismatch: requested={requested}, stored={stored:?}")]
    RowIdentityMismatch {
        requested: EpisodeInstanceId,
        stored: String,
    },
    #[error("persisted escrow blob size is invalid: actual={actual}, max={max}")]
    InvalidEscrowSize { actual: usize, max: usize },
    #[error("persisted escrow payload identity mismatch: requested={requested}, payload={payload}")]
    PayloadIdentityMismatch {
        requested: EpisodeInstanceId,
        payload: EpisodeInstanceId,
    },
    #[error("persisted escrow digest must be exactly 32 bytes; actual={actual}")]
    InvalidDigestLength { actual: usize },
    #[error("persisted escrow digest may not be zero")]
    ZeroDigest,
    #[error("persisted escrow digest does not match its encoded payload")]
    DigestMismatch,
    #[error("persisted escrow persistence reference is invalid")]
    InvalidPersistenceReference,
    #[error("could not decode persisted escrow: {0}")]
    Decoding(String),
    #[error("persisted escrow validation failed: {0}")]
    EscrowValidation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
    use symthaea_welfare_assurance::memory_identity::episode_content_id;
    use symthaea_welfare_assurance::memory_quarantine::{
        EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrowPersistence,
        episodic_instance_target_id,
    };

    fn persisted_fixture(
        path: &Path,
    ) -> (
        EpisodeInstanceId,
        Sha256Digest,
        String,
        EpisodicQuarantineEscrow,
    ) {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig {
            psi_threshold: 0.0,
            ..Default::default()
        });
        let id = memory
            .store_if_significant_with_id(Episode::new(
                ContinuousHV::from_vec(vec![0.1; 8]),
                ContinuousHV::from_vec(vec![0.9; 8]),
                0.8,
                10,
            ))
            .unwrap();
        let episode = memory
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .unwrap()
            .1;
        let target = episodic_instance_target_id("symthaea:self:episodic-memory", id).unwrap();
        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target,
            instance_id: id,
            content_id: episode_content_id(&episode).unwrap(),
            captured_at_unix_s: 20,
            pre_active_state_digest: Sha256Digest([7; 32]),
            episode,
        };
        let digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let mut store = SqliteEpisodicContinuityStore::open(path).unwrap();
        let reference = store
            .persist_episodic_quarantine_escrow(&escrow, digest)
            .unwrap();
        (id, digest, reference, escrow)
    }

    #[test]
    fn exact_point_lookup_returns_only_requested_valid_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("continuity.sqlite");
        let (id, digest, reference, escrow) = persisted_fixture(&path);
        let lookup = SqliteEpisodicEscrowLookup::open(&path).unwrap();
        let row = lookup
            .load_episodic_quarantine_escrow(id)
            .unwrap()
            .unwrap();
        assert_eq!(row.escrow.instance_id, id);
        assert_eq!(row.escrow.content_id, escrow.content_id);
        assert_eq!(row.stored_digest, digest);
        assert_eq!(row.persistence_ref, reference);
    }

    #[test]
    fn unknown_occurrence_returns_none_without_enumeration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("continuity.sqlite");
        let (id, _, _, _) = persisted_fixture(&path);
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig {
            psi_threshold: 0.0,
            ..Default::default()
        });
        let other = memory
            .store_if_significant_with_id(Episode::new(
                ContinuousHV::from_vec(vec![0.2; 8]),
                ContinuousHV::from_vec(vec![0.8; 8]),
                0.7,
                11,
            ))
            .unwrap();
        assert_ne!(id, other);
        let lookup = SqliteEpisodicEscrowLookup::open(&path).unwrap();
        assert!(lookup.load_episodic_quarantine_escrow(other).unwrap().is_none());
    }

    #[test]
    fn tampered_digest_fails_before_row_leaves_lookup_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("continuity.sqlite");
        let (id, _, _, _) = persisted_fixture(&path);
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE quarantine_escrows SET escrow_digest = ?1 WHERE instance_id = ?2",
            params![vec![9u8; 32], id.to_string()],
        )
        .unwrap();
        drop(conn);

        let lookup = SqliteEpisodicEscrowLookup::open(&path).unwrap();
        assert!(matches!(
            lookup.load_episodic_quarantine_escrow(id),
            Err(SqliteEpisodicEscrowLookupError::DigestMismatch)
        ));
    }

    #[test]
    fn wrong_database_schema_is_rejected_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("continuity.sqlite");
        let _ = persisted_fixture(&path);
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE continuity_meta SET value = 'wrong-schema' WHERE key = 'schema'",
            [],
        )
        .unwrap();
        drop(conn);
        assert!(matches!(
            SqliteEpisodicEscrowLookup::open(&path),
            Err(SqliteEpisodicEscrowLookupError::UnsupportedDatabaseSchema(_))
        ));
    }
}
