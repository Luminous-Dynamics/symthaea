// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SQLite-backed implementation of the Agency Kernel checkpoint CAS contract.
//!
//! The security property is the transaction boundary, not SQLite as a brand:
//! every writer uses `BEGIN IMMEDIATE`, compares the exact durable frontier, and
//! installs one successor inside the same transaction. A stale process therefore
//! cannot successfully publish a second successor from an already-consumed head.
//! `synchronous=FULL` is required, but actual power-loss durability still depends
//! on the filesystem/storage stack honoring SQLite's sync requests.

#![deny(unsafe_code)]

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_authority_frontier::CheckpointCasStore;
use thiserror::Error;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS agency_checkpoint_frontier (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    digest BLOB NOT NULL CHECK (length(digest) = 32),
    checkpoint BLOB NOT NULL
);
"#;

/// Durable single-frontier SQLite store.
pub struct SqliteCheckpointCasStore {
    connection: Connection,
}

impl SqliteCheckpointCasStore {
    /// Open/create a store and apply the durability/concurrency profile used by
    /// this implementation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteFrontierError> {
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;\n\
             PRAGMA synchronous=FULL;\n\
             PRAGMA foreign_keys=ON;\n\
             PRAGMA busy_timeout=5000;",
        )?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self { connection })
    }

    /// Open an in-memory store. Useful for deterministic tests only; it does not
    /// provide crash/power-loss durability.
    pub fn open_in_memory() -> Result<Self, SqliteFrontierError> {
        let connection = Connection::open_in_memory()?;
        connection.execute_batch(
            "PRAGMA synchronous=FULL;\n\
             PRAGMA foreign_keys=ON;\n\
             PRAGMA busy_timeout=5000;",
        )?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self { connection })
    }

    /// Read and integrity-check the current durable frontier.
    pub fn load_frontier(
        &self,
    ) -> Result<Option<(GrantAccountCheckpoint, CheckpointHead)>, SqliteFrontierError> {
        read_frontier_from_connection(&self.connection)
    }
}

impl CheckpointCasStore for SqliteCheckpointCasStore {
    type Error = SqliteFrontierError;

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHead>,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
        let next_head = checkpoint
            .head()
            .map_err(|_| SqliteFrontierError::CheckpointEncoding)?;
        let checkpoint_bytes =
            bincode::serialize(checkpoint).map_err(|_| SqliteFrontierError::CheckpointEncoding)?;
        let next_sequence = i64::try_from(next_head.sequence)
            .map_err(|_| SqliteFrontierError::SequenceOutOfRange)?;

        // BEGIN IMMEDIATE acquires SQLite's write reservation before the read.
        // The comparison and write are therefore one serialized state transition,
        // not an unlocked read followed by an independent write.
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = read_frontier_from_transaction(&transaction)?;

        if current.as_ref().map(|(_, head)| *head) != expected_previous {
            return Err(SqliteFrontierError::Conflict {
                expected: expected_previous,
                actual: current.map(|(_, head)| head),
            });
        }

        match expected_previous {
            None => {
                let inserted = transaction.execute(
                    "INSERT INTO agency_checkpoint_frontier(singleton, sequence, digest, checkpoint)\n\
                     VALUES(1, ?1, ?2, ?3)",
                    params![next_sequence, &next_head.digest.0[..], checkpoint_bytes],
                )?;
                if inserted != 1 {
                    return Err(SqliteFrontierError::AtomicWriteInvariant);
                }
            }
            Some(previous) => {
                let previous_sequence = i64::try_from(previous.sequence)
                    .map_err(|_| SqliteFrontierError::SequenceOutOfRange)?;
                let updated = transaction.execute(
                    "UPDATE agency_checkpoint_frontier\n\
                     SET sequence = ?1, digest = ?2, checkpoint = ?3\n\
                     WHERE singleton = 1 AND sequence = ?4 AND digest = ?5",
                    params![
                        next_sequence,
                        &next_head.digest.0[..],
                        checkpoint_bytes,
                        previous_sequence,
                        &previous.digest.0[..]
                    ],
                )?;
                if updated != 1 {
                    return Err(SqliteFrontierError::Conflict {
                        expected: Some(previous),
                        actual: read_frontier_from_transaction(&transaction)?
                            .map(|(_, head)| head),
                    });
                }
            }
        }

        transaction.commit()?;

        // Re-read through SQLite after commit instead of trusting the in-memory
        // candidate. This catches codec/row corruption before returning success.
        let (_, durable_head) = self
            .load_frontier()?
            .ok_or(SqliteFrontierError::AtomicWriteInvariant)?;
        if durable_head != next_head {
            return Err(SqliteFrontierError::DurableHeadMismatch);
        }
        Ok(next_head)
    }
}

fn read_frontier_from_connection(
    connection: &Connection,
) -> Result<Option<(GrantAccountCheckpoint, CheckpointHead)>, SqliteFrontierError> {
    connection
        .query_row(
            "SELECT sequence, digest, checkpoint FROM agency_checkpoint_frontier WHERE singleton = 1",
            [],
            decode_row,
        )
        .optional()?
        .map(validate_stored_frontier)
        .transpose()
}

fn read_frontier_from_transaction(
    transaction: &Transaction<'_>,
) -> Result<Option<(GrantAccountCheckpoint, CheckpointHead)>, SqliteFrontierError> {
    transaction
        .query_row(
            "SELECT sequence, digest, checkpoint FROM agency_checkpoint_frontier WHERE singleton = 1",
            [],
            decode_row,
        )
        .optional()?
        .map(validate_stored_frontier)
        .transpose()
}

fn decode_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<(i64, Vec<u8>, Vec<u8>)> {
    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
}

fn validate_stored_frontier(
    row: (i64, Vec<u8>, Vec<u8>),
) -> Result<(GrantAccountCheckpoint, CheckpointHead), SqliteFrontierError> {
    let (sequence, digest, checkpoint_bytes) = row;
    let sequence = u64::try_from(sequence).map_err(|_| SqliteFrontierError::CorruptRow)?;
    let digest: [u8; 32] = digest
        .try_into()
        .map_err(|_| SqliteFrontierError::CorruptRow)?;
    let checkpoint: GrantAccountCheckpoint = bincode::deserialize(&checkpoint_bytes)
        .map_err(|_| SqliteFrontierError::CorruptCheckpoint)?;
    let computed = checkpoint
        .head()
        .map_err(|_| SqliteFrontierError::CorruptCheckpoint)?;
    let stored = CheckpointHead {
        sequence,
        digest: symthaea_authority::Digest32(digest),
    };
    if computed != stored {
        return Err(SqliteFrontierError::CorruptCheckpoint);
    }
    Ok((checkpoint, stored))
}

/// SQLite CAS/durability failures.
#[derive(Debug, Error)]
pub enum SqliteFrontierError {
    #[error("SQLite frontier operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("checkpoint encoding failed")]
    CheckpointEncoding,
    #[error("checkpoint sequence cannot be represented by SQLite INTEGER")]
    SequenceOutOfRange,
    #[error("stored frontier row is malformed")]
    CorruptRow,
    #[error("stored checkpoint bytes do not match the stored frontier head")]
    CorruptCheckpoint,
    #[error("checkpoint CAS conflict: expected {expected:?}, actual {actual:?}")]
    Conflict {
        expected: Option<CheckpointHead>,
        actual: Option<CheckpointHead>,
    },
    #[error("SQLite CAS write did not affect exactly one frontier row")]
    AtomicWriteInvariant,
    #[error("durable frontier after commit does not equal the requested checkpoint")]
    DurableHeadMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use symthaea_action_runtime::GrantAccount;
    use symthaea_authority::{AuthorityEpoch, CapabilityGrant, PrincipalId, RiskBudget};
    use symthaea_authority_frontier::establish_grant_frontier;

    static NEXT_DB: AtomicU64 = AtomicU64::new(0);

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "sqlite-cas-test",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            AuthorityEpoch(1),
        );
        grant.max_uses = 2;
        grant.risk_budget = RiskBudget {
            mutation_units: 2,
            ..RiskBudget::default()
        };
        grant
    }

    fn db_path() -> std::path::PathBuf {
        let id = NEXT_DB.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "symthaea-agency-cas-{}-{id}.sqlite",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{}-wal", path.display()));
        let _ = fs::remove_file(format!("{}-shm", path.display()));
    }

    #[test]
    fn committed_frontier_survives_reopen_and_revalidates() {
        let path = db_path();
        cleanup(&path);
        let grant = grant();
        let store = SqliteCheckpointCasStore::open(&path).unwrap();
        let (frontier, _adapter) = establish_grant_frontier(&grant, store).unwrap();

        let reopened = SqliteCheckpointCasStore::open(&path).unwrap();
        let (checkpoint, head) = reopened.load_frontier().unwrap().unwrap();
        assert_eq!(head, frontier.head);
        assert_eq!(checkpoint, frontier.checkpoint);
        cleanup(&path);
    }

    #[test]
    fn two_connections_cannot_both_advance_the_same_head() {
        let path = db_path();
        cleanup(&path);
        let grant = grant();
        let mut first = SqliteCheckpointCasStore::open(&path).unwrap();
        let account = GrantAccount::new(&grant);
        let genesis = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        let genesis_head = first.compare_and_swap(None, &genesis).unwrap();

        let mut writer_a = SqliteCheckpointCasStore::open(&path).unwrap();
        let mut writer_b = SqliteCheckpointCasStore::open(&path).unwrap();
        let successor = GrantAccountCheckpoint::successor(&genesis, &grant, account.snapshot()).unwrap();
        let winner = writer_a
            .compare_and_swap(Some(genesis_head), &successor)
            .unwrap();
        assert_eq!(winner, successor.head().unwrap());

        assert!(matches!(
            writer_b.compare_and_swap(Some(genesis_head), &successor),
            Err(SqliteFrontierError::Conflict { .. })
        ));
        cleanup(&path);
    }

    #[test]
    fn wrong_expected_head_never_mutates_the_database() {
        let path = db_path();
        cleanup(&path);
        let grant = grant();
        let mut store = SqliteCheckpointCasStore::open(&path).unwrap();
        let account = GrantAccount::new(&grant);
        let genesis = GrantAccountCheckpoint::first(&grant, account.snapshot()).unwrap();
        let actual = store.compare_and_swap(None, &genesis).unwrap();
        let successor = GrantAccountCheckpoint::successor(&genesis, &grant, account.snapshot()).unwrap();
        let wrong = CheckpointHead {
            sequence: actual.sequence,
            digest: symthaea_authority::Digest32([0xA5; 32]),
        };
        assert!(matches!(
            store.compare_and_swap(Some(wrong), &successor),
            Err(SqliteFrontierError::Conflict { .. })
        ));
        assert_eq!(store.load_frontier().unwrap().unwrap().1, actual);
        cleanup(&path);
    }
}
