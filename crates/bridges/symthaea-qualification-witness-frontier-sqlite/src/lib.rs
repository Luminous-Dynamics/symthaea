// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concrete SQLite history adapter for ancestry-aware witness-frontier recovery.
//!
//! The guard deliberately trades write availability for a strong point-in-time
//! claim. It first asks #449's sequence store to perform the authoritative full
//! chain audit, then acquires `BEGIN IMMEDIATE`, proves the frontier has not
//! changed since that audit, snapshots only sequence -> reservation-head history,
//! and keeps the SQLite writer reservation until the guard is dropped/released.
//!
//! This prevents a local writer from advancing the witness frontier between
//! recovery classification and an external publication/anchoring decision.
//! The adapter does not reimplement reservation-digest semantics.

#![deny(unsafe_code)]

use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use symthaea_authority::Digest32;
use symthaea_qualification_witness_frontier::{
    classify_witness_frontier_recovery_v1, FrontierHistoryError, FrontierRecoveryError,
    LocalWitnessFrontierHistory, VerifiedExternalWitnessFrontierV1, WitnessFrontierPointV1,
    WitnessFrontierPublicationDispositionV1, WitnessFrontierRecoveryRelationV1,
};
use symthaea_qualification_witness_sequence::{
    SqliteWitnessSequenceStore, WitnessSequenceFrontierV1,
};
use thiserror::Error;

const SQLITE_APPLICATION_ID: i64 = 1_398_363_953; // ASCII "SYW1"
const SQLITE_USER_VERSION: i64 = 1;
const ZERO32: [u8; 32] = [0; 32];

/// A writer-barrier-backed, immutable view of one audited witness history.
///
/// While this value exists, another #449 writer using `BEGIN IMMEDIATE` cannot
/// reserve or persist witness state in the same SQLite database. Drop releases
/// the barrier with `ROLLBACK`; `release` releases it explicitly.
#[derive(Debug)]
pub struct SqliteWitnessFrontierPublicationGuard {
    connection: Option<Connection>,
    witness_id: [u8; 16],
    current: Option<WitnessFrontierPointV1>,
    historical_heads: Vec<Digest32>,
}

impl SqliteWitnessFrontierPublicationGuard {
    pub fn acquire(
        store: &SqliteWitnessSequenceStore,
        witness_id: [u8; 16],
    ) -> Result<Self, SqliteWitnessFrontierGuardError> {
        if witness_id == [0; 16] {
            return Err(SqliteWitnessFrontierGuardError::InvalidWitnessId);
        }

        // #449 remains the authority for complete chain/state validation.
        let audited = store.audit_witness(witness_id)?;
        let audited_statement = store.frontier_statement(witness_id)?;
        match (audited, audited_statement) {
            (None, None) => {}
            (Some(frontier), Some(statement))
                if statement.witness_id() == witness_id
                    && statement.high_watermark() == frontier.high_watermark
                    && statement.reservation_head() == frontier.reservation_head => {}
            _ => return Err(SqliteWitnessFrontierGuardError::AuditStatementMismatch),
        }

        let mut connection = open_guard_connection(store.path())?;
        connection.execute_batch("BEGIN IMMEDIATE;")?;

        let result = (|| {
            verify_database_identity(&connection)?;
            let locked_frontier = load_frontier(&connection, witness_id)?;
            if locked_frontier != audited {
                return Err(SqliteWitnessFrontierGuardError::FrontierChangedDuringAcquisition);
            }

            let historical_heads = load_historical_heads(&connection, witness_id)?;
            match locked_frontier {
                None => {
                    if !historical_heads.is_empty() {
                        return Err(SqliteWitnessFrontierGuardError::HistoricalStateMismatch);
                    }
                }
                Some(frontier) => {
                    let count = u64::try_from(historical_heads.len())
                        .map_err(|_| SqliteWitnessFrontierGuardError::HistoricalStateMismatch)?;
                    if count != frontier.high_watermark
                        || historical_heads.last().copied() != Some(frontier.reservation_head)
                    {
                        return Err(SqliteWitnessFrontierGuardError::HistoricalStateMismatch);
                    }
                }
            }

            let current = audited_statement.map(WitnessFrontierPointV1::from_local_statement);
            Ok((current, historical_heads))
        })();

        match result {
            Ok((current, historical_heads)) => Ok(Self {
                connection: Some(connection),
                witness_id,
                current,
                historical_heads,
            }),
            Err(error) => {
                let _ = connection.execute_batch("ROLLBACK;");
                Err(error)
            }
        }
    }

    pub fn witness_id(&self) -> [u8; 16] {
        self.witness_id
    }

    pub fn current_frontier_point(&self) -> Option<WitnessFrontierPointV1> {
        self.current
    }

    pub fn historical_head(&self, sequence: u64) -> Option<Digest32> {
        if sequence == 0 {
            return None;
        }
        usize::try_from(sequence - 1)
            .ok()
            .and_then(|index| self.historical_heads.get(index).copied())
    }

    /// Classify against an already authenticated/current-enough external anchor.
    ///
    /// The SQLite writer barrier remains held after this returns. Callers that
    /// perform anchoring/publication should keep the guard alive until that
    /// operation's local-frontier precondition has been consumed.
    pub fn classify(
        &self,
        external: Option<&VerifiedExternalWitnessFrontierV1>,
    ) -> Result<GuardedWitnessFrontierDecisionV1<'_>, SqliteWitnessFrontierGuardError> {
        let relation = classify_witness_frontier_recovery_v1(self, self.witness_id, external)?;
        Ok(GuardedWitnessFrontierDecisionV1 {
            guard: self,
            relation,
        })
    }

    /// Explicitly release the writer barrier. No local state is mutated by the
    /// guard, so `ROLLBACK` is intentional and equivalent to commit for data.
    pub fn release(mut self) -> Result<(), SqliteWitnessFrontierGuardError> {
        if let Some(connection) = self.connection.take() {
            connection.execute_batch("ROLLBACK;")?;
        }
        Ok(())
    }
}

impl Drop for SqliteWitnessFrontierPublicationGuard {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            let _ = connection.execute_batch("ROLLBACK;");
        }
    }
}

/// A classification that borrows the publication guard, making the lock
/// lifetime visible in the type system while the decision is inspected.
#[derive(Debug)]
pub struct GuardedWitnessFrontierDecisionV1<'a> {
    guard: &'a SqliteWitnessFrontierPublicationGuard,
    relation: WitnessFrontierRecoveryRelationV1,
}

impl GuardedWitnessFrontierDecisionV1<'_> {
    pub fn relation(&self) -> WitnessFrontierRecoveryRelationV1 {
        self.relation
    }

    pub fn publication_disposition(&self) -> WitnessFrontierPublicationDispositionV1 {
        self.relation.publication_disposition()
    }

    pub fn witness_id(&self) -> [u8; 16] {
        self.guard.witness_id
    }

    pub fn local_frontier(&self) -> Option<WitnessFrontierPointV1> {
        self.guard.current
    }
}

impl LocalWitnessFrontierHistory for SqliteWitnessFrontierPublicationGuard {
    fn audit_witness(&self, witness_id: [u8; 16]) -> Result<(), FrontierHistoryError> {
        if witness_id != self.witness_id {
            return Err(history_error("witness id does not match guarded history"));
        }
        // The full #449 audit completed before the writer barrier was acquired,
        // and acquisition proved the locked current frontier still matched it.
        Ok(())
    }

    fn current_frontier(
        &self,
        witness_id: [u8; 16],
    ) -> Result<Option<WitnessFrontierPointV1>, FrontierHistoryError> {
        if witness_id != self.witness_id {
            return Err(history_error("witness id does not match guarded history"));
        }
        Ok(self.current)
    }

    fn reservation_head_at(
        &self,
        witness_id: [u8; 16],
        high_watermark: u64,
    ) -> Result<Option<Digest32>, FrontierHistoryError> {
        if witness_id != self.witness_id {
            return Err(history_error("witness id does not match guarded history"));
        }
        Ok(self.historical_head(high_watermark))
    }
}

fn open_guard_connection(path: &Path) -> Result<Connection, SqliteWitnessFrontierGuardError> {
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_NOFOLLOW
        | OpenFlags::SQLITE_OPEN_EXRESCODE;
    let connection = Connection::open_with_flags(path, flags)?;
    connection.busy_timeout(Duration::from_secs(10))?;
    Ok(connection)
}

fn verify_database_identity(
    connection: &Connection,
) -> Result<(), SqliteWitnessFrontierGuardError> {
    let application_id: i64 = connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    let user_version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if application_id != SQLITE_APPLICATION_ID || user_version != SQLITE_USER_VERSION {
        return Err(SqliteWitnessFrontierGuardError::SchemaIdentityMismatch);
    }

    let mut statement = connection.prepare(
        "SELECT type, name FROM sqlite_schema\n\
         WHERE name NOT LIKE 'sqlite_%' ORDER BY type ASC, name ASC",
    )?;
    let objects = statement
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let expected = vec![
        ("table".to_string(), "witness_sequence_attempts".to_string()),
        ("table".to_string(), "witness_sequence_frontier".to_string()),
    ];
    if objects != expected {
        return Err(SqliteWitnessFrontierGuardError::SchemaIdentityMismatch);
    }
    Ok(())
}

fn load_frontier(
    connection: &Connection,
    witness_id: [u8; 16],
) -> Result<Option<WitnessSequenceFrontierV1>, SqliteWitnessFrontierGuardError> {
    connection
        .query_row(
            "SELECT high_watermark, reservation_head\n\
             FROM witness_sequence_frontier WHERE witness_id=?1",
            params![&witness_id[..]],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()?
        .map(|(high_watermark, head)| {
            let high_watermark = u64::try_from(high_watermark)
                .map_err(|_| SqliteWitnessFrontierGuardError::HistoricalStateMismatch)?;
            let reservation_head = Digest32(exact_array::<32>(&head)?);
            if high_watermark == 0 || reservation_head.0 == ZERO32 {
                return Err(SqliteWitnessFrontierGuardError::HistoricalStateMismatch);
            }
            Ok(WitnessSequenceFrontierV1 {
                high_watermark,
                reservation_head,
            })
        })
        .transpose()
}

fn load_historical_heads(
    connection: &Connection,
    witness_id: [u8; 16],
) -> Result<Vec<Digest32>, SqliteWitnessFrontierGuardError> {
    let mut statement = connection.prepare(
        "SELECT sequence, reservation_digest\n\
         FROM witness_sequence_attempts\n\
         WHERE witness_id=?1 ORDER BY sequence ASC",
    )?;
    let mut rows = statement.query(params![&witness_id[..]])?;
    let mut expected_sequence = 1u64;
    let mut heads = Vec::new();
    while let Some(row) = rows.next()? {
        let raw_sequence: i64 = row.get(0)?;
        let raw_head: Vec<u8> = row.get(1)?;
        let sequence = u64::try_from(raw_sequence)
            .map_err(|_| SqliteWitnessFrontierGuardError::HistoricalStateMismatch)?;
        let head = Digest32(exact_array::<32>(&raw_head)?);
        if sequence != expected_sequence || head.0 == ZERO32 {
            return Err(SqliteWitnessFrontierGuardError::HistoricalStateMismatch);
        }
        heads.push(head);
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or(SqliteWitnessFrontierGuardError::HistoricalStateMismatch)?;
    }
    Ok(heads)
}

fn exact_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], SqliteWitnessFrontierGuardError> {
    bytes
        .try_into()
        .map_err(|_| SqliteWitnessFrontierGuardError::HistoricalStateMismatch)
}

fn history_error(reason: &str) -> FrontierHistoryError {
    FrontierHistoryError {
        reason: reason.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum SqliteWitnessFrontierGuardError {
    #[error("invalid witness id")]
    InvalidWitnessId,
    #[error("#449 audit and frontier statement disagreed")]
    AuditStatementMismatch,
    #[error("SQLite witness-sequence schema/application identity mismatch")]
    SchemaIdentityMismatch,
    #[error("witness frontier changed between full audit and writer-barrier acquisition")]
    FrontierChangedDuringAcquisition,
    #[error("guarded historical witness state is malformed or disagrees with the audited frontier")]
    HistoricalStateMismatch,
    #[error(transparent)]
    Recovery(#[from] FrontierRecoveryError),
    #[error(transparent)]
    Sequence(#[from] symthaea_qualification_witness_sequence::WitnessSequenceError),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use symthaea_qualification_witness_frontier::{
        verify_external_witness_frontier_v1, ExternalAnchorVerificationError,
        ExternalWitnessFrontierClaimV1, ExternalWitnessFrontierVerifier,
        EXTERNAL_ANCHOR_SCHEMA_VERSION,
    };
    use symthaea_qualification_witness_sequence::WitnessSequenceAttemptBindingV1;

    fn db_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "symthaea-frontier-guard-{name}-{}-{}.sqlite3",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(format!("{}-wal", path.display()));
        let _ = fs::remove_file(format!("{}-shm", path.display()));
        path
    }

    fn binding(attempt: u8) -> WitnessSequenceAttemptBindingV1 {
        WitnessSequenceAttemptBindingV1 {
            attempt_id: [attempt; 16],
            witness_id: [0x51; 16],
            witness_epoch: 7,
            archive_sha256: Digest32([0x11; 32]),
            git_head: [0x22; 20],
            git_tree: [0x33; 20],
            verifier_digest: Digest32([0x44; 32]),
            witness_policy_digest: Digest32([0x55; 32]),
        }
    }

    struct AcceptExternal;

    impl ExternalWitnessFrontierVerifier for AcceptExternal {
        fn verify_current(
            &self,
            _claim: &ExternalWitnessFrontierClaimV1,
        ) -> Result<(), ExternalAnchorVerificationError> {
            Ok(())
        }
    }

    fn claim_from_statement(
        statement: symthaea_qualification_witness_sequence::WitnessSequenceFrontierStatementV1,
    ) -> ExternalWitnessFrontierClaimV1 {
        ExternalWitnessFrontierClaimV1 {
            schema_version: EXTERNAL_ANCHOR_SCHEMA_VERSION,
            source_id: [0x61; 16],
            source_epoch: 3,
            source_sequence: 9,
            witness_id: statement.witness_id(),
            high_watermark: statement.high_watermark(),
            reservation_head: statement.reservation_head(),
            frontier_statement_digest: statement.digest(),
            freshness_evidence_digest: Digest32([0x77; 32]),
        }
    }

    #[test]
    fn guarded_history_proves_trusted_prefix_ancestry() {
        let path = db_path("ancestry");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding(1)).unwrap();
        store.reserve_attempt(binding(2)).unwrap();
        let trusted_statement = store.frontier_statement([0x51; 16]).unwrap().unwrap();
        store.reserve_attempt(binding(3)).unwrap();

        let external = verify_external_witness_frontier_v1(
            claim_from_statement(trusted_statement),
            &AcceptExternal,
        )
        .unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(Some(&external)).unwrap();
        assert!(matches!(
            decision.relation(),
            WitnessFrontierRecoveryRelationV1::LocalAheadVerifiedDescendant {
                trusted_high_watermark: 2,
                local_high_watermark: 3,
                ..
            }
        ));
        assert_eq!(
            decision.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::AnchorRequired
        );
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn exact_current_anchor_is_publishable_while_guard_is_held() {
        let path = db_path("current");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding(1)).unwrap();
        let statement = store.frontier_statement([0x51; 16]).unwrap().unwrap();
        let external = verify_external_witness_frontier_v1(
            claim_from_statement(statement),
            &AcceptExternal,
        )
        .unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(Some(&external)).unwrap();
        assert_eq!(
            decision.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::PublishAllowed
        );
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn guard_blocks_new_sequence_writer_until_release() {
        let path = db_path("writer-block");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding(1)).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();

        let writer = SqliteWitnessSequenceStore::open(&path).unwrap();
        let (tx, rx) = mpsc::channel();
        let thread = thread::spawn(move || {
            let result = writer.reserve_attempt(binding(2)).map(|reservation| reservation.sequence);
            tx.send(result).unwrap();
        });

        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        guard.release().unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap(), 2);
        thread.join().unwrap();
        cleanup(&path);
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{}-wal", path.display()));
        let _ = fs::remove_file(format!("{}-shm", path.display()));
    }
}
