//! Durable local witness journal for CIV governance checkpoints.
//!
//! This crate persists accepted history and prepared transitions in SQLite.
//! Anti-rollback requires an independently operated implementation of
//! IndependentAnchor; this crate intentionally does not provide one.
//!
//! The store uses WAL + synchronous=FULL and verifies its runtime PRAGMAs on
//! every opened connection. This profile depends on SQLite's VFS and the
//! underlying filesystem honoring sync requests. It is not a claim against an
//! attacker who can roll back both this database and the external anchor.

use rusqlite::{
    params, Connection, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub type Digest = [u8; 32];

const PROTOCOL_VERSION: u16 = 1;
const SCHEMA_VERSION: i64 = 1;
const RECORD_DOMAIN: &[u8] = b"mycelix-civ013-durable-record-v1\0";
const FORK_DOMAIN: &[u8] = b"mycelix-civ013-durable-fork-v1\0";
const ZERO_DIGEST: Digest = [0; 32];
const BUSY_TIMEOUT_MS: u64 = 5_000;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS witness_meta (
    log_id TEXT PRIMARY KEY NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    generation INTEGER NOT NULL CHECK (generation > 0),
    record_digest BLOB NOT NULL CHECK (length(record_digest) = 32)
);
CREATE TABLE IF NOT EXISTS witness_records (
    log_id TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation > 0),
    record_digest BLOB NOT NULL CHECK (length(record_digest) = 32),
    protocol_version INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    anchor_digest BLOB NOT NULL CHECK (length(anchor_digest) = 32),
    receipt_sequence INTEGER NOT NULL CHECK (receipt_sequence >= 0),
    receipt_digest BLOB,
    previous_record_digest BLOB,
    status INTEGER NOT NULL CHECK (status IN (0, 1)),
    PRIMARY KEY (log_id, generation),
    UNIQUE (log_id, record_digest),
    CHECK ((receipt_sequence = 0 AND receipt_digest IS NULL)
        OR (receipt_sequence > 0 AND receipt_digest IS NOT NULL)),
    CHECK (receipt_digest IS NULL OR length(receipt_digest) = 32),
    CHECK (previous_record_digest IS NULL OR length(previous_record_digest) = 32)
);
CREATE TABLE IF NOT EXISTS witness_fork_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation > 0),
    first_record_digest BLOB NOT NULL CHECK (length(first_record_digest) = 32),
    conflicting_record_digest BLOB NOT NULL CHECK (length(conflicting_record_digest) = 32),
    previous_evidence_digest BLOB,
    digest BLOB NOT NULL CHECK (length(digest) = 32),
    CHECK (first_record_digest != conflicting_record_digest),
    CHECK (previous_evidence_digest IS NULL OR length(previous_evidence_digest) = 32),
    UNIQUE (log_id, digest)
);
CREATE TABLE IF NOT EXISTS witness_fork_meta (
    log_id TEXT PRIMARY KEY NOT NULL,
    evidence_count INTEGER NOT NULL CHECK (evidence_count > 0),
    tail_digest BLOB NOT NULL CHECK (length(tail_digest) = 32)
);
CREATE INDEX IF NOT EXISTS witness_records_status_idx
    ON witness_records(log_id, status, generation);
";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Record {
    pub protocol_version: u16,
    pub generation: u64,
    pub log_id: String,
    pub policy_version: String,
    pub anchor_digest: Digest,
    pub receipt_sequence: u64,
    pub receipt_digest: Option<Digest>,
    pub previous_record_digest: Option<Digest>,
    pub digest: Digest,
}

impl Record {
    fn build(
        generation: u64,
        log_id: &str,
        policy_version: &str,
        anchor_digest: Digest,
        receipt_sequence: u64,
        receipt_digest: Option<Digest>,
        previous_record_digest: Option<Digest>,
    ) -> Self {
        let digest = record_digest(
            PROTOCOL_VERSION,
            generation,
            log_id,
            policy_version,
            anchor_digest,
            receipt_sequence,
            receipt_digest,
            previous_record_digest,
        );
        Self {
            protocol_version: PROTOCOL_VERSION,
            generation,
            log_id: log_id.to_owned(),
            policy_version: policy_version.to_owned(),
            anchor_digest,
            receipt_sequence,
            receipt_digest,
            previous_record_digest,
            digest,
        }
    }

    fn is_self_consistent(&self) -> bool {
        self.protocol_version == PROTOCOL_VERSION
            && !self.log_id.is_empty()
            && !self.policy_version.is_empty()
            && valid_receipt_tail(self.receipt_sequence, self.receipt_digest)
            && self.digest
                == record_digest(
                    self.protocol_version,
                    self.generation,
                    &self.log_id,
                    &self.policy_version,
                    self.anchor_digest,
                    self.receipt_sequence,
                    self.receipt_digest,
                    self.previous_record_digest,
                )
    }
}

/// Externally retained anchor position. Generation zero with a zero digest is
/// the provisioned, empty-log bootstrap state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnchorState {
    pub log_id: String,
    pub generation: u64,
    pub record_digest: Digest,
}

impl AnchorState {
    pub fn genesis(log_id: impl Into<String>) -> Self {
        Self {
            log_id: log_id.into(),
            generation: 0,
            record_digest: ZERO_DIGEST,
        }
    }

    fn from_record(record: &Record) -> Self {
        Self {
            log_id: record.log_id.clone(),
            generation: record.generation,
            record_digest: record.digest,
        }
    }
}

/// Implement this interface in a separate trust/failure domain.
/// Implementations must provide durable reads and a linearizable CAS. Do not
/// point this interface back at the same SQLite database.
pub trait IndependentAnchor: Send + Sync {
    fn current(&self, log_id: &str) -> Result<AnchorState, AnchorError>;

    fn compare_and_advance(
        &self,
        expected: &AnchorState,
        next: &AnchorState,
    ) -> Result<(), AnchorError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnchorError {
    Unavailable,
    CompareFailed,
    Other(String),
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "independent anchor unavailable"),
            Self::CompareFailed => write!(f, "independent anchor compare-and-advance failed"),
            Self::Other(message) => write!(f, "independent anchor error: {message}"),
        }
    }
}

impl std::error::Error for AnchorError {}

#[derive(Debug)]
pub enum WitnessError {
    Sqlite(rusqlite::Error),
    Io(std::io::Error),
    Anchor(AnchorError),
    InvalidInput(&'static str),
    RuntimeConfigurationMismatch,
    CorruptStore(&'static str),
    CorruptForkEvidence,
    StalePredecessor,
    ReceiptRollback,
    ReceiptTailEquivocation,
    ExternalAnchorMismatch,
    RollbackDetected,
    PreparedCandidateConflict,
    GenerationOverflow,
    InjectedCrash(FaultPoint),
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(error) => write!(f, "SQLite error: {error}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Anchor(error) => write!(f, "{error}"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::RuntimeConfigurationMismatch => {
                write!(f, "SQLite durability PRAGMAs do not match the required profile")
            }
            Self::CorruptStore(message) => write!(f, "corrupt witness store: {message}"),
            Self::CorruptForkEvidence => write!(f, "corrupt witness fork-evidence chain"),
            Self::StalePredecessor => write!(f, "stale predecessor"),
            Self::ReceiptRollback => write!(f, "receipt sequence rollback"),
            Self::ReceiptTailEquivocation => {
                write!(f, "receipt digest changed without a sequence advance")
            }
            Self::ExternalAnchorMismatch => {
                write!(f, "external anchor disagrees with local history")
            }
            Self::RollbackDetected => {
                write!(f, "external anchor proves local rollback or missing history")
            }
            Self::PreparedCandidateConflict => {
                write!(f, "a different candidate is already prepared for this generation")
            }
            Self::GenerationOverflow => write!(f, "witness generation overflow"),
            Self::InjectedCrash(point) => write!(f, "injected crash at {point:?}"),
        }
    }
}

impl std::error::Error for WitnessError {}

impl From<rusqlite::Error> for WitnessError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value)
    }
}

impl From<std::io::Error> for WitnessError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<AnchorError> for WitnessError {
    fn from(value: AnchorError) -> Self {
        Self::Anchor(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaultPoint {
    AfterPrepare,
    AfterExternalAnchorAdvance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct History {
    accepted: Option<Record>,
    prepared: Option<Record>,
}

pub struct SqliteWitnessStore {
    path: PathBuf,
}

impl SqliteWitnessStore {
    /// Open or initialize an on-disk journal with WAL + FULL durability.
    /// In-memory SQLite is unsupported because operations open independent
    /// connections for cross-connection serialization tests.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WitnessError> {
        let path = path.as_ref().to_path_buf();
        if path.as_os_str().is_empty() {
            return Err(WitnessError::InvalidInput("database path is empty"));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let store = Self { path };
        let conn = store.open_connection()?;
        conn.execute_batch(SCHEMA)?;
        store.integrity_check()?;
        Ok(store)
    }

    fn open_connection(&self) -> Result<Connection, WitnessError> {
        let conn = Connection::open(&self.path)?;
        conn.busy_timeout(Duration::from_millis(BUSY_TIMEOUT_MS))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             PRAGMA busy_timeout=5000;",
        )?;

        let journal_mode: String =
            conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        let synchronous: i64 =
            conn.query_row("PRAGMA synchronous", [], |row| row.get(0))?;
        let foreign_keys: i64 =
            conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
        let busy_timeout: i64 =
            conn.query_row("PRAGMA busy_timeout", [], |row| row.get(0))?;
        if !journal_mode.eq_ignore_ascii_case("wal")
            || synchronous != 2
            || foreign_keys != 1
            || busy_timeout != BUSY_TIMEOUT_MS as i64
        {
            return Err(WitnessError::RuntimeConfigurationMismatch);
        }
        Ok(conn)
    }

    /// Initialize from the provisioned genesis position (generation zero,
    /// zero digest). The caller must authenticate and authorize the initial
    /// policy version and `anchor_digest` before invoking this method: this
    /// crate does not authenticate the caller or establish the semantic truth
    /// of the first checkpoint. The independent anchor protects subsequent
    /// history only if it is operated outside the database's rollback domain.
    /// Generation one is externally anchored before local acceptance.
    /// Repeated identical requests are idempotent.
    pub fn initialize(
        &self,
        log_id: &str,
        policy_version: &str,
        anchor_digest: Digest,
        receipt_sequence: u64,
        receipt_digest: Option<Digest>,
        anchor: &dyn IndependentAnchor,
    ) -> Result<Record, WitnessError> {
        self.transition_inner(
            log_id,
            0,
            ZERO_DIGEST,
            Some(policy_version),
            anchor_digest,
            receipt_sequence,
            receipt_digest,
            anchor,
            None,
        )
    }

    /// Propose, prepare, externally anchor, and locally accept the next record.
    /// The returned record is not a digital signature or witness quorum proof.
    pub fn advance(
        &self,
        log_id: &str,
        expected_generation: u64,
        expected_digest: Digest,
        proposed_anchor_digest: Digest,
        receipt_sequence: u64,
        receipt_digest: Option<Digest>,
        anchor: &dyn IndependentAnchor,
    ) -> Result<Record, WitnessError> {
        self.transition_inner(
            log_id,
            expected_generation,
            expected_digest,
            None,
            proposed_anchor_digest,
            receipt_sequence,
            receipt_digest,
            anchor,
            None,
        )
    }

    fn transition_inner(
        &self,
        log_id: &str,
        expected_generation: u64,
        expected_digest: Digest,
        initial_policy: Option<&str>,
        proposed_anchor_digest: Digest,
        receipt_sequence: u64,
        receipt_digest: Option<Digest>,
        anchor: &dyn IndependentAnchor,
        fault: Option<FaultPoint>,
    ) -> Result<Record, WitnessError> {
        validate_input(log_id, initial_policy, receipt_sequence, receipt_digest)?;
        let current = self.recover(log_id, anchor)?;

        if let Some(current) = current.as_ref() {
            let next_generation = expected_generation
                .checked_add(1)
                .ok_or(WitnessError::GenerationOverflow)?;
            if current.generation == next_generation
                && current.previous_record_digest
                    == (expected_generation > 0).then_some(expected_digest)
                && initial_policy.map_or(true, |policy| current.policy_version == policy)
                && current.anchor_digest == proposed_anchor_digest
                && current.receipt_sequence == receipt_sequence
                && current.receipt_digest == receipt_digest
            {
                return Ok(current.clone());
            }
            if current.generation != expected_generation || current.digest != expected_digest {
                return Err(WitnessError::StalePredecessor);
            }
        } else if expected_generation != 0 || expected_digest != ZERO_DIGEST {
            return Err(WitnessError::StalePredecessor);
        }

        let policy_version = current
            .as_ref()
            .map(|record| record.policy_version.as_str())
            .or(initial_policy)
            .ok_or(WitnessError::InvalidInput(
                "first transition requires a policy version",
            ))?;
        if let Some(previous) = current.as_ref() {
            if receipt_sequence < previous.receipt_sequence {
                return Err(WitnessError::ReceiptRollback);
            }
            if receipt_sequence == previous.receipt_sequence
                && receipt_digest != previous.receipt_digest
            {
                return Err(WitnessError::ReceiptTailEquivocation);
            }
        }
        let generation = expected_generation
            .checked_add(1)
            .ok_or(WitnessError::GenerationOverflow)?;
        if generation > i64::MAX as u64 {
            return Err(WitnessError::GenerationOverflow);
        }
        let candidate = Record::build(
            generation,
            log_id,
            policy_version,
            proposed_anchor_digest,
            receipt_sequence,
            receipt_digest,
            (expected_generation > 0).then_some(expected_digest),
        );
        let expected_anchor = if let Some(previous) = current.as_ref() {
            AnchorState::from_record(previous)
        } else {
            AnchorState::genesis(log_id)
        };
        let observed = anchor.current(log_id)?;
        if observed != expected_anchor && observed != AnchorState::from_record(&candidate) {
            // Fork evidence is only valid for two competing records in the same
            // log and generation. A later anchor position proves that this local
            // candidate is stale/missing from accepted history; its digest must
            // never be mislabeled as a same-generation competitor.
            if observed.log_id == log_id && observed.generation == generation {
                self.record_fork(log_id, generation, observed.record_digest, candidate.digest)?;
                return Err(WitnessError::ExternalAnchorMismatch);
            }
            if observed.log_id == log_id && observed.generation > generation {
                return Err(WitnessError::RollbackDetected);
            }
            return Err(WitnessError::ExternalAnchorMismatch);
        }

        self.prepare(&candidate, expected_generation, expected_digest)?;
        if fault == Some(FaultPoint::AfterPrepare) {
            return Err(WitnessError::InjectedCrash(FaultPoint::AfterPrepare));
        }

        let observed_after_prepare = anchor.current(log_id)?;
        if observed_after_prepare == expected_anchor {
            let next_anchor = AnchorState::from_record(&candidate);
            if let Err(cas_error) = anchor.compare_and_advance(&expected_anchor, &next_anchor) {
                let observed_after_cas = anchor.current(log_id)?;
                if observed_after_cas != next_anchor {
                    if observed_after_cas.log_id == log_id
                        && observed_after_cas.generation == generation
                        && observed_after_cas.record_digest != candidate.digest
                    {
                        self.record_fork(
                            log_id,
                            generation,
                            observed_after_cas.record_digest,
                            candidate.digest,
                        )?;
                        return Err(WitnessError::ExternalAnchorMismatch);
                    }
                    if observed_after_cas.log_id != log_id {
                        return Err(WitnessError::ExternalAnchorMismatch);
                    }
                    if observed_after_cas.generation > generation {
                        return Err(WitnessError::RollbackDetected);
                    }
                    return Err(WitnessError::Anchor(cas_error));
                }
            }
        } else if observed_after_prepare != AnchorState::from_record(&candidate) {
            if observed_after_prepare.log_id == log_id
                && observed_after_prepare.generation == generation
            {
                self.record_fork(
                    log_id,
                    generation,
                    observed_after_prepare.record_digest,
                    candidate.digest,
                )?;
                return Err(WitnessError::ExternalAnchorMismatch);
            }
            if observed_after_prepare.log_id == log_id
                && observed_after_prepare.generation > generation
            {
                return Err(WitnessError::RollbackDetected);
            }
            return Err(WitnessError::ExternalAnchorMismatch);
        }
        if fault == Some(FaultPoint::AfterExternalAnchorAdvance) {
            return Err(WitnessError::InjectedCrash(
                FaultPoint::AfterExternalAnchorAdvance,
            ));
        }

        self.finalize(&candidate, expected_generation, expected_digest)?;
        Ok(candidate)
    }

    /// Recover only when local accepted history agrees with the independent
    /// anchor, or when the anchor matches an exact prepared one-step successor.
    pub fn recover(
        &self,
        log_id: &str,
        anchor: &dyn IndependentAnchor,
    ) -> Result<Option<Record>, WitnessError> {
        if log_id.is_empty() {
            return Err(WitnessError::InvalidInput("log ID is empty"));
        }
        let history = self.load_history(log_id)?;
        let external = anchor.current(log_id)?;
        if external.log_id != log_id {
            return Err(WitnessError::ExternalAnchorMismatch);
        }

        let accepted = history.accepted.as_ref();
        let expected = accepted
            .map(AnchorState::from_record)
            .unwrap_or_else(|| AnchorState::genesis(log_id));

        if external == expected {
            return Ok(history.accepted);
        }
        if external.generation < expected.generation {
            return Err(WitnessError::RollbackDetected);
        }
        if external.generation == expected.generation {
            return Err(WitnessError::ExternalAnchorMismatch);
        }

        if let Some(prepared) = history.prepared.as_ref() {
            let is_exact_successor = prepared.generation == expected.generation + 1
                && prepared.previous_record_digest
                    == (expected.generation > 0).then_some(expected.record_digest)
                && external == AnchorState::from_record(prepared);
            if is_exact_successor {
                self.finalize(
                    prepared,
                    expected.generation,
                    expected.record_digest,
                )?;
                return Ok(Some(prepared.clone()));
            }
        }
        Err(WitnessError::RollbackDetected)
    }

    fn prepare(
        &self,
        candidate: &Record,
        expected_generation: u64,
        expected_digest: Digest,
    ) -> Result<(), WitnessError> {
        let mut conn = self.open_connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let meta: Option<(i64, Vec<u8>)> = tx
            .query_row(
                "SELECT generation, record_digest FROM witness_meta WHERE log_id=?1",
                params![candidate.log_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        match (expected_generation, meta) {
            (0, None) => {}
            (generation, Some((stored_generation, stored_digest)))
                if stored_generation == generation as i64
                    && blob_digest(&stored_digest)? == expected_digest => {}
            _ => return Err(WitnessError::StalePredecessor),
        }

        let existing: Option<(Vec<u8>, i64)> = tx
            .query_row(
                "SELECT record_digest, status FROM witness_records
                 WHERE log_id=?1 AND generation=?2",
                params![candidate.log_id, candidate.generation as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((stored_digest, status)) = existing {
            let stored_digest = blob_digest(&stored_digest)?;
            if stored_digest == candidate.digest && status == 0 {
                tx.commit()?;
                return Ok(());
            }
            if stored_digest != candidate.digest {
                append_fork_evidence(
                    &tx,
                    &candidate.log_id,
                    candidate.generation,
                    stored_digest,
                    candidate.digest,
                )?;
                tx.commit()?;
                return Err(WitnessError::PreparedCandidateConflict);
            }
            return Err(WitnessError::CorruptStore(
                "candidate digest exists with an invalid status",
            ));
        }

        tx.execute(
            "INSERT INTO witness_records (
                log_id, generation, record_digest, protocol_version, policy_version,
                anchor_digest, receipt_sequence, receipt_digest, previous_record_digest, status
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0)",
            params![
                candidate.log_id,
                candidate.generation as i64,
                candidate.digest.as_slice(),
                candidate.protocol_version as i64,
                candidate.policy_version,
                candidate.anchor_digest.as_slice(),
                candidate.receipt_sequence as i64,
                candidate.receipt_digest.map(|digest| digest.to_vec()),
                candidate.previous_record_digest.map(|digest| digest.to_vec()),
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn finalize(
        &self,
        candidate: &Record,
        expected_generation: u64,
        expected_digest: Digest,
    ) -> Result<(), WitnessError> {
        let mut conn = self.open_connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let meta: Option<(i64, Vec<u8>)> = tx
            .query_row(
                "SELECT generation, record_digest FROM witness_meta WHERE log_id=?1",
                params![candidate.log_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        if let Some((generation, digest)) = meta.as_ref() {
            if *generation == candidate.generation as i64
                && blob_digest(digest)? == candidate.digest
            {
                let status: i64 = tx.query_row(
                    "SELECT status FROM witness_records
                     WHERE log_id=?1 AND generation=?2 AND record_digest=?3",
                    params![
                        candidate.log_id,
                        candidate.generation as i64,
                        candidate.digest.as_slice()
                    ],
                    |row| row.get(0),
                )?;
                if status == 1 {
                    let expected_candidate_generation = expected_generation
                        .checked_add(1)
                        .ok_or(WitnessError::GenerationOverflow)?;
                    if candidate.generation != expected_candidate_generation
                        || candidate.previous_record_digest
                            != (expected_generation > 0).then_some(expected_digest)
                        || !candidate.is_self_consistent()
                    {
                        return Err(WitnessError::StalePredecessor);
                    }
                    // An equal metadata pointer and digest column do not prove the
                    // current record fields are intact. Validate the complete
                    // history before treating a retry as idempotent success.
                    let validated_history =
                        Self::load_history_from_connection(&tx, &candidate.log_id)?;
                    if !validated_history.accepted.as_ref().is_some_and(|head| {
                        head.generation == candidate.generation
                            && head.digest == candidate.digest
                    }) {
                        return Err(WitnessError::CorruptStore(
                            "current candidate does not match accepted history",
                        ));
                    }
                    tx.commit()?;
                    return Ok(());
                }
            }

            // Recovery may have finalized this candidate and a later writer may
            // already have advanced the head before the original caller resumes.
            // Validate the current head digest's encoded shape before using this
            // branch; an accepted historical row is not permission to overlook
            // malformed current metadata.
            if *generation > candidate.generation as i64 {
                let current_head_digest = blob_digest(digest)?;
                // A well-shaped 32-byte metadata pointer is not enough. Prove that
                // it still names an accepted record at the current generation before
                // returning idempotent success for a historical candidate.
                let current_head_row: Option<(Vec<u8>, i64)> = tx
                    .query_row(
                        "SELECT record_digest, status FROM witness_records
                         WHERE log_id=?1 AND generation=?2",
                        params![candidate.log_id, *generation],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let Some((stored_head_digest, current_head_status)) = current_head_row else {
                    return Err(WitnessError::CorruptStore("metadata head record is missing"));
                };
                if current_head_status != 1
                    || blob_digest(&stored_head_digest)? != current_head_digest
                {
                    return Err(WitnessError::CorruptStore(
                        "metadata record digest does not match current accepted head",
                    ));
                }
                // The metadata pointer can match its row digest while the record
                // fields themselves have been corrupted. Validate full history in
                // the same transaction snapshot before allowing late-finalize success.
                let validated_history =
                    Self::load_history_from_connection(&tx, &candidate.log_id)?;
                if !validated_history.accepted.as_ref().is_some_and(|head| {
                    head.generation == *generation as u64 && head.digest == current_head_digest
                }) {
                    return Err(WitnessError::CorruptStore(
                        "metadata does not match accepted history",
                    ));
                }

                // Return idempotent success only if this exact candidate is still a
                // committed historical record, is self-consistent, and is the exact
                // successor named by the caller's predecessor. Never move the head
                // backwards or treat a different same-generation candidate as success.
                let expected_candidate_generation = expected_generation
                    .checked_add(1)
                    .ok_or(WitnessError::GenerationOverflow)?;
                if candidate.generation == expected_candidate_generation
                    && candidate.previous_record_digest
                        == (expected_generation > 0).then_some(expected_digest)
                    && candidate.is_self_consistent()
                {
                    let status: Option<i64> = tx
                        .query_row(
                            "SELECT status FROM witness_records
                             WHERE log_id=?1 AND generation=?2 AND record_digest=?3",
                            params![
                                candidate.log_id,
                                candidate.generation as i64,
                                candidate.digest.as_slice()
                            ],
                            |row| row.get(0),
                        )
                        .optional()?;
                    if status == Some(1) {
                        tx.commit()?;
                        return Ok(());
                    }
                }
                return Err(WitnessError::StalePredecessor);
            }
        }

        match (expected_generation, meta) {
            (0, None) if candidate.generation == 1 => {}
            (generation, Some((stored_generation, stored_digest)))
                if stored_generation == generation as i64
                    && blob_digest(&stored_digest)? == expected_digest => {}
            _ => return Err(WitnessError::StalePredecessor),
        }

        let affected = tx.execute(
            "UPDATE witness_records SET status=1
             WHERE log_id=?1 AND generation=?2 AND record_digest=?3 AND status=0",
            params![
                candidate.log_id,
                candidate.generation as i64,
                candidate.digest.as_slice()
            ],
        )?;
        if affected != 1 {
            return Err(WitnessError::CorruptStore(
                "prepared candidate missing during finalization",
            ));
        }
        if expected_generation == 0 {
            tx.execute(
                "INSERT INTO witness_meta (log_id, schema_version, generation, record_digest)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    candidate.log_id,
                    SCHEMA_VERSION,
                    candidate.generation as i64,
                    candidate.digest.as_slice()
                ],
            )?;
        } else {
            let affected = tx.execute(
                "UPDATE witness_meta SET generation=?1, record_digest=?2
                 WHERE log_id=?3 AND generation=?4 AND record_digest=?5",
                params![
                    candidate.generation as i64,
                    candidate.digest.as_slice(),
                    candidate.log_id,
                    expected_generation as i64,
                    expected_digest.as_slice()
                ],
            )?;
            if affected != 1 {
                return Err(WitnessError::StalePredecessor);
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn record_fork(
        &self,
        log_id: &str,
        generation: u64,
        first_digest: Digest,
        conflicting_digest: Digest,
    ) -> Result<(), WitnessError> {
        if first_digest == conflicting_digest {
            return Err(WitnessError::InvalidInput(
                "fork evidence must contain distinct record digests",
            ));
        }
        let mut conn = self.open_connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        append_fork_evidence(
            &tx,
            log_id,
            generation,
            first_digest,
            conflicting_digest,
        )?;
        tx.commit()?;
        Ok(())
    }

    fn load_history_from_connection(
        conn: &Connection,
        log_id: &str,
    ) -> Result<History, WitnessError> {
        validate_fork_history(conn, log_id)?;

        let mut statement = conn.prepare(
            "SELECT generation, record_digest, protocol_version, policy_version,
                    anchor_digest, receipt_sequence, receipt_digest,
                    previous_record_digest, status
             FROM witness_records WHERE log_id=?1 ORDER BY generation ASC",
        )?;
        let raw_rows = statement
            .query_map(params![log_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, Option<Vec<u8>>>(6)?,
                    row.get::<_, Option<Vec<u8>>>(7)?,
                    row.get::<_, i64>(8)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);

        let meta: Option<(i64, Vec<u8>, i64)> = conn
            .query_row(
                "SELECT generation, record_digest, schema_version
                 FROM witness_meta WHERE log_id=?1",
                params![log_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        let mut records = Vec::with_capacity(raw_rows.len());
        let mut statuses = Vec::with_capacity(raw_rows.len());
        let mut previous: Option<Record> = None;
        for raw in raw_rows {
            let (generation, digest, protocol_version, policy_version, anchor_digest, sequence,
                receipt_digest, previous_digest, status) = raw;
            if generation <= 0 || sequence < 0 || !(status == 0 || status == 1) {
                return Err(WitnessError::CorruptStore("invalid numeric record field"));
            }
            let record = Record {
                protocol_version: u16::try_from(protocol_version)
                    .map_err(|_| WitnessError::CorruptStore("invalid protocol version"))?,
                generation: generation as u64,
                log_id: log_id.to_owned(),
                policy_version,
                anchor_digest: blob_digest(&anchor_digest)?,
                receipt_sequence: sequence as u64,
                receipt_digest: receipt_digest.as_deref().map(blob_digest).transpose()?,
                previous_record_digest: previous_digest
                    .as_deref()
                    .map(blob_digest)
                    .transpose()?,
                digest: blob_digest(&digest)?,
            };
            let expected_generation = records.len() as u64 + 1;
            let expected_previous = previous.as_ref().map(|record| record.digest);
            if record.generation != expected_generation
                || record.previous_record_digest != expected_previous
                || !record.is_self_consistent()
                || previous.as_ref().is_some_and(|prior| {
                    prior.policy_version != record.policy_version
                        || record.receipt_sequence < prior.receipt_sequence
                        || (record.receipt_sequence == prior.receipt_sequence
                            && record.receipt_digest != prior.receipt_digest)
                })
            {
                return Err(WitnessError::CorruptStore(
                    "record chain, policy, or receipt history is invalid",
                ));
            }
            previous = Some(record.clone());
            records.push(record);
            statuses.push(status);
        }

        let accepted_count = statuses.iter().filter(|status| **status == 1).count();
        let prepared_count = statuses.iter().filter(|status| **status == 0).count();
        if prepared_count > 1 {
            return Err(WitnessError::CorruptStore(
                "more than one prepared candidate exists",
            ));
        }
        if let Some(position) = statuses.iter().position(|status| *status == 0) {
            if position + 1 != statuses.len()
                || statuses[..position].iter().any(|status| *status != 1)
            {
                return Err(WitnessError::CorruptStore(
                    "prepared candidate is not the final record",
                ));
            }
        }
        if statuses.iter().skip(accepted_count).any(|status| *status != 0) {
            return Err(WitnessError::CorruptStore("accepted record appears after prepared state"));
        }

        let accepted = match meta {
            Some((generation, digest, schema_version)) => {
                if schema_version != SCHEMA_VERSION
                    || generation <= 0
                    || generation as usize != accepted_count
                    || accepted_count == 0
                    || blob_digest(&digest)? != records[accepted_count - 1].digest
                {
                    return Err(WitnessError::CorruptStore(
                        "metadata does not match accepted history",
                    ));
                }
                Some(records[accepted_count - 1].clone())
            }
            None if accepted_count == 0 => None,
            None => return Err(WitnessError::CorruptStore("accepted records lack metadata")),
        };
        let prepared = records
            .get(accepted_count)
            .filter(|_| prepared_count == 1)
            .cloned();

        if records.len() != accepted_count + prepared_count {
            return Err(WitnessError::CorruptStore("record status accounting mismatch"));
        }

        Ok(History { accepted, prepared })
    }

    fn load_history(&self, log_id: &str) -> Result<History, WitnessError> {
        let mut conn = self.open_connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let history = Self::load_history_from_connection(&tx, log_id)?;
        tx.commit()?;
        Ok(history)
    }

    /// Run SQLite integrity checks and return "ok" on success. This does not
    /// validate the external anchor.
    pub fn integrity_check(&self) -> Result<String, WitnessError> {
        let conn = self.open_connection()?;
        let result: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        if result != "ok" {
            return Err(WitnessError::CorruptStore("SQLite integrity_check failed"));
        }
        let foreign_key_violation: Option<String> = conn
            .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
            .optional()?;
        if foreign_key_violation.is_some() {
            return Err(WitnessError::CorruptStore("SQLite foreign_key_check failed"));
        }
        Ok(result)
    }
}

fn append_fork_evidence(
    tx: &Transaction<'_>,
    log_id: &str,
    generation: u64,
    first_digest: Digest,
    conflicting_digest: Digest,
) -> Result<(), WitnessError> {
    if first_digest == conflicting_digest {
        return Err(WitnessError::InvalidInput(
            "fork evidence must contain distinct record digests",
        ));
    }

    let row_count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM witness_fork_evidence WHERE log_id=?1",
        params![log_id],
        |row| row.get(0),
    )?;
    let previous_blob: Option<Vec<u8>> = tx
        .query_row(
            "SELECT digest FROM witness_fork_evidence
             WHERE log_id=?1 ORDER BY id DESC LIMIT 1",
            params![log_id],
            |row| row.get(0),
        )
        .optional()?;
    let previous = previous_blob.as_deref().map(blob_digest).transpose()?;
    let prior_meta: Option<(i64, Vec<u8>)> = tx
        .query_row(
            "SELECT evidence_count, tail_digest FROM witness_fork_meta WHERE log_id=?1",
            params![log_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;

    match (row_count, previous, prior_meta) {
        (0, None, None) => {}
        (count, Some(tail), Some((meta_count, meta_tail)))
            if count > 0
                && meta_count == count
                && blob_digest(&meta_tail)
                    .map_err(|_| WitnessError::CorruptForkEvidence)? == tail => {}
        _ => return Err(WitnessError::CorruptForkEvidence),
    }

    let next_count = row_count
        .checked_add(1)
        .ok_or(WitnessError::CorruptForkEvidence)?;
    let digest = fork_digest(log_id, generation, first_digest, conflicting_digest, previous);
    tx.execute(
        "INSERT INTO witness_fork_evidence (
            log_id, generation, first_record_digest, conflicting_record_digest,
            previous_evidence_digest, digest
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            log_id,
            generation as i64,
            first_digest.as_slice(),
            conflicting_digest.as_slice(),
            previous.map(|value| value.to_vec()),
            digest.as_slice()
        ],
    )?;
    tx.execute(
        "INSERT INTO witness_fork_meta (log_id, evidence_count, tail_digest)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(log_id) DO UPDATE SET
            evidence_count=excluded.evidence_count,
            tail_digest=excluded.tail_digest",
        params![log_id, next_count, digest.as_slice()],
    )?;
    Ok(())
}

fn validate_fork_history(conn: &Connection, log_id: &str) -> Result<(), WitnessError> {
    let mut statement = conn.prepare(
        "SELECT generation, first_record_digest, conflicting_record_digest,
                previous_evidence_digest, digest
         FROM witness_fork_evidence WHERE log_id=?1 ORDER BY id ASC",
    )?;
    let rows = statement
        .query_map(params![log_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Option<Vec<u8>>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(statement);

    let mut previous: Option<Digest> = None;
    for (generation, first, conflicting, prev, digest) in &rows {
        if *generation <= 0 {
            return Err(WitnessError::CorruptForkEvidence);
        }
        let first = blob_digest(first).map_err(|_| WitnessError::CorruptForkEvidence)?;
        let conflicting =
            blob_digest(conflicting).map_err(|_| WitnessError::CorruptForkEvidence)?;
        let stored_previous = prev
            .as_deref()
            .map(blob_digest)
            .transpose()
            .map_err(|_| WitnessError::CorruptForkEvidence)?;
        let stored_digest =
            blob_digest(digest).map_err(|_| WitnessError::CorruptForkEvidence)?;
        if first == conflicting
            || stored_previous != previous
            || stored_digest
                != fork_digest(log_id, *generation as u64, first, conflicting, previous)
        {
            return Err(WitnessError::CorruptForkEvidence);
        }
        previous = Some(stored_digest);
    }

    let stored_meta: Option<(i64, Vec<u8>)> = conn
        .query_row(
            "SELECT evidence_count, tail_digest FROM witness_fork_meta WHERE log_id=?1",
            params![log_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let actual_count =
        i64::try_from(rows.len()).map_err(|_| WitnessError::CorruptForkEvidence)?;
    match (actual_count, previous, stored_meta) {
        (0, None, None) => Ok(()),
        (count, Some(tail), Some((meta_count, meta_tail)))
            if count > 0
                && meta_count == count
                && blob_digest(&meta_tail)
                    .map_err(|_| WitnessError::CorruptForkEvidence)? == tail => Ok(()),
        _ => Err(WitnessError::CorruptForkEvidence),
    }
}

fn valid_receipt_tail(sequence: u64, digest: Option<Digest>) -> bool {
    (sequence == 0) == digest.is_none()
}

fn validate_input(
    log_id: &str,
    policy: Option<&str>,
    sequence: u64,
    digest: Option<Digest>,
) -> Result<(), WitnessError> {
    if log_id.is_empty() {
        return Err(WitnessError::InvalidInput("log ID is empty"));
    }
    if policy.is_some_and(str::is_empty) {
        return Err(WitnessError::InvalidInput("policy version is empty"));
    }
    if !valid_receipt_tail(sequence, digest) {
        return Err(WitnessError::InvalidInput("receipt sequence/digest shape is invalid"));
    }
    if sequence > i64::MAX as u64 {
        return Err(WitnessError::InvalidInput("receipt sequence exceeds SQLite INTEGER range"));
    }
    Ok(())
}

fn encode_field(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

fn option_digest(out: &mut Vec<u8>, value: Option<Digest>) {
    match value {
        Some(digest) => {
            out.push(1);
            out.extend_from_slice(&digest);
        }
        None => out.push(0),
    }
}

fn record_digest(
    protocol_version: u16,
    generation: u64,
    log_id: &str,
    policy_version: &str,
    anchor_digest: Digest,
    receipt_sequence: u64,
    receipt_digest: Option<Digest>,
    previous_record_digest: Option<Digest>,
) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(RECORD_DOMAIN);
    bytes.extend_from_slice(&protocol_version.to_be_bytes());
    bytes.extend_from_slice(&generation.to_be_bytes());
    encode_field(&mut bytes, log_id.as_bytes());
    encode_field(&mut bytes, policy_version.as_bytes());
    bytes.extend_from_slice(&anchor_digest);
    bytes.extend_from_slice(&receipt_sequence.to_be_bytes());
    option_digest(&mut bytes, receipt_digest);
    option_digest(&mut bytes, previous_record_digest);
    Sha256::digest(bytes).into()
}

fn fork_digest(
    log_id: &str,
    generation: u64,
    first: Digest,
    conflicting: Digest,
    previous: Option<Digest>,
) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(FORK_DOMAIN);
    encode_field(&mut bytes, log_id.as_bytes());
    bytes.extend_from_slice(&generation.to_be_bytes());
    bytes.extend_from_slice(&first);
    bytes.extend_from_slice(&conflicting);
    option_digest(&mut bytes, previous);
    Sha256::digest(bytes).into()
}

fn blob_digest(bytes: &[u8]) -> Result<Digest, WitnessError> {
    bytes
        .try_into()
        .map_err(|_| WitnessError::CorruptStore("digest blob is not 32 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    static NEXT_DB: AtomicU64 = AtomicU64::new(1);

    struct TempDb(PathBuf);

    impl TempDb {
        fn new() -> Self {
            let id = NEXT_DB.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "symthaea-civ-witness-{}-{id}.sqlite",
                std::process::id()
            )))
        }

        fn open(&self) -> SqliteWitnessStore {
            SqliteWitnessStore::open(&self.0).expect("open SQLite witness store")
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let _ = std::fs::remove_file(self.0.with_extension("sqlite-wal"));
            let _ = std::fs::remove_file(self.0.with_extension("sqlite-shm"));
        }
    }

    #[derive(Default)]
    struct MemoryAnchor {
        states: Mutex<HashMap<String, AnchorState>>,
    }

    impl MemoryAnchor {
        fn provision(&self, log_id: &str) {
            self.states
                .lock()
                .expect("anchor lock")
                .insert(log_id.to_owned(), AnchorState::genesis(log_id));
        }
    }


    struct RacingAnchor {
        inner: MemoryAnchor,
        race_next: Mutex<bool>,
    }

    impl RacingAnchor {
        fn new(log_id: &str) -> Self {
            let anchor = Self {
                inner: MemoryAnchor::default(),
                race_next: Mutex::new(false),
            };
            anchor.inner.provision(log_id);
            anchor
        }

        fn arm_race(&self) {
            *self.race_next.lock().expect("race flag lock") = true;
        }
    }

    impl IndependentAnchor for RacingAnchor {
        fn current(&self, log_id: &str) -> Result<AnchorState, AnchorError> {
            self.inner.current(log_id)
        }

        fn compare_and_advance(
            &self,
            expected: &AnchorState,
            next: &AnchorState,
        ) -> Result<(), AnchorError> {
            let mut race = self
                .race_next
                .lock()
                .map_err(|_| AnchorError::Other("race flag poisoned".into()))?;
            if *race {
                *race = false;
                drop(race);
                let competitor = AnchorState {
                    log_id: next.log_id.clone(),
                    generation: next.generation,
                    record_digest: h(b"competing-external-record"),
                };
                self.inner
                    .states
                    .lock()
                    .map_err(|_| AnchorError::Other("anchor mutex poisoned".into()))?
                    .insert(competitor.log_id.clone(), competitor);
                return Err(AnchorError::CompareFailed);
            }
            self.inner.compare_and_advance(expected, next)
        }
    }


    struct AdvanceDuringReadAnchor {
        inner: MemoryAnchor,
        calls: AtomicU64,
        advance_on_call: u64,
        advanced_state: AnchorState,
    }

    impl AdvanceDuringReadAnchor {
        fn new(log_id: &str, initial: AnchorState, advance_on_call: u64, advanced_state: AnchorState) -> Self {
            let inner = MemoryAnchor::default();
            inner.states.lock().expect("anchor lock").insert(log_id.to_owned(), initial);
            Self { inner, calls: AtomicU64::new(0), advance_on_call, advanced_state }
        }
    }

    impl IndependentAnchor for AdvanceDuringReadAnchor {
        fn current(&self, log_id: &str) -> Result<AnchorState, AnchorError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == self.advance_on_call {
                self.inner.states.lock().map_err(|_| AnchorError::Other("anchor mutex poisoned".into()))?
                    .insert(log_id.to_owned(), self.advanced_state.clone());
            }
            self.inner.current(log_id)
        }

        fn compare_and_advance(&self, expected: &AnchorState, next: &AnchorState) -> Result<(), AnchorError> {
            self.inner.compare_and_advance(expected, next)
        }
    }

    struct OneShotUnavailableAnchor {
        inner: MemoryAnchor,
        calls: AtomicU64,
        fail_on_call: AtomicU64,
    }

    impl OneShotUnavailableAnchor {
        fn new() -> Self {
            Self {
                inner: MemoryAnchor::default(),
                calls: AtomicU64::new(0),
                fail_on_call: AtomicU64::new(0),
            }
        }

        fn provision(&self, log_id: &str) {
            self.inner.provision(log_id);
        }

        fn fail_on_nth_next_current(&self, call: u64) {
            self.calls.store(0, Ordering::SeqCst);
            self.fail_on_call.store(call, Ordering::SeqCst);
        }
    }

    impl IndependentAnchor for OneShotUnavailableAnchor {
        fn current(&self, log_id: &str) -> Result<AnchorState, AnchorError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self
                .fail_on_call
                .compare_exchange(call, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Err(AnchorError::Unavailable);
            }
            self.inner.current(log_id)
        }

        fn compare_and_advance(
            &self,
            expected: &AnchorState,
            next: &AnchorState,
        ) -> Result<(), AnchorError> {
            self.inner.compare_and_advance(expected, next)
        }
    }

    impl IndependentAnchor for MemoryAnchor {
        fn current(&self, log_id: &str) -> Result<AnchorState, AnchorError> {
            self.states
                .lock()
                .map_err(|_| AnchorError::Other("anchor mutex poisoned".into()))?
                .get(log_id)
                .cloned()
                .ok_or(AnchorError::Unavailable)
        }

        fn compare_and_advance(
            &self,
            expected: &AnchorState,
            next: &AnchorState,
        ) -> Result<(), AnchorError> {
            let mut states = self
                .states
                .lock()
                .map_err(|_| AnchorError::Other("anchor mutex poisoned".into()))?;
            let current = states
                .get(&expected.log_id)
                .ok_or(AnchorError::Unavailable)?;
            if current != expected
                || next.log_id != expected.log_id
                || next.generation != expected.generation + 1
            {
                return Err(AnchorError::CompareFailed);
            }
            states.insert(next.log_id.clone(), next.clone());
            Ok(())
        }
    }

    fn h(value: &[u8]) -> Digest {
        Sha256::digest(value).into()
    }

    fn initialize(
        store: &SqliteWitnessStore,
        anchor: &MemoryAnchor,
        log_id: &str,
    ) -> Record {
        anchor.provision(log_id);
        store
            .initialize(log_id, "policy-v1", h(b"genesis-checkpoint"), 0, None, anchor)
            .expect("initialize trusted witness")
    }

    #[test]
    fn uses_wal_full_and_recovers_after_reopen() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-a");
        let repeated_initialization = store
            .initialize(
                "log-a",
                "policy-v1",
                h(b"genesis-checkpoint"),
                0,
                None,
                &anchor,
            )
            .expect("identical initialization is idempotent");
        assert_eq!(repeated_initialization, first);
        let next = store
            .advance(
                "log-a",
                first.generation,
                first.digest,
                h(b"checkpoint-two"),
                1,
                Some(h(b"receipt-one")),
                &anchor,
            )
            .expect("advance witness");
        assert_eq!(next.generation, 2);
        assert_eq!(store.integrity_check().expect("integrity"), "ok");

        drop(store);
        let reopened = db.open();
        assert_eq!(reopened.recover("log-a", &anchor).expect("recovery"), Some(next));
    }

    #[test]
    fn receipt_tail_cannot_regress_or_equivocate() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-tail");
        let second = store
            .advance(
                "log-tail",
                first.generation,
                first.digest,
                h(b"checkpoint-two"),
                2,
                Some(h(b"receipt-two")),
                &anchor,
            )
            .expect("second record");

        assert!(matches!(
            store.advance(
                "log-tail",
                second.generation,
                second.digest,
                h(b"checkpoint-three"),
                1,
                Some(h(b"receipt-old")),
                &anchor,
            ),
            Err(WitnessError::ReceiptRollback)
        ));
        assert!(matches!(
            store.advance(
                "log-tail",
                second.generation,
                second.digest,
                h(b"checkpoint-three"),
                2,
                Some(h(b"equivocated-receipt")),
                &anchor,
            ),
            Err(WitnessError::ReceiptTailEquivocation)
        ));
        assert_eq!(store.recover("log-tail", &anchor).expect("unchanged history"), Some(second));
    }

    #[test]
    fn crash_after_external_advance_is_reconciled_from_prepared_record() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-reconcile");
        let result = store.transition_inner(
            "log-reconcile",
            first.generation,
            first.digest,
            None,
            h(b"checkpoint-two"),
            1,
            Some(h(b"receipt-one")),
            &anchor,
            Some(FaultPoint::AfterExternalAnchorAdvance),
        );
        assert!(matches!(
            result,
            Err(WitnessError::InjectedCrash(FaultPoint::AfterExternalAnchorAdvance))
        ));
        drop(store);

        let reopened = db.open();
        let recovered = reopened.recover("log-reconcile", &anchor)
            .expect("prepared candidate recovery")
            .expect("accepted recovery");
        assert_eq!(recovered.generation, 2);
        assert_eq!(recovered.previous_record_digest, Some(first.digest));
    }

    #[test]
    fn initialize_fails_closed_when_external_anchor_is_unavailable() {
        let db = TempDb::new();
        let store = db.open();
        let anchor = MemoryAnchor::default();

        assert!(matches!(
            store.initialize(
                "log-unavailable-anchor",
                "policy-v1",
                h(b"genesis-checkpoint"),
                0,
                None,
                &anchor,
            ),
            Err(WitnessError::Anchor(AnchorError::Unavailable))
        ));
        assert!(matches!(
            store.load_history("log-unavailable-anchor"),
            Ok(History {
                accepted: None,
                prepared: None
            })
        ));
        assert_eq!(store.integrity_check().expect("integrity"), "ok");
    }

    #[test]
    fn initialize_rejects_anchor_state_bound_to_another_log() {
        let db = TempDb::new();
        let store = db.open();
        let anchor = MemoryAnchor::default();
        anchor.provision("log-wrong-anchor");
        anchor
            .states
            .lock()
            .expect("anchor lock")
            .get_mut("log-wrong-anchor")
            .expect("provisioned state")
            .log_id = "different-log-id".to_owned();

        assert!(matches!(
            store.initialize(
                "log-wrong-anchor",
                "policy-v1",
                h(b"genesis-checkpoint"),
                0,
                None,
                &anchor,
            ),
            Err(WitnessError::ExternalAnchorMismatch)
        ));
        assert!(matches!(
            store.load_history("log-wrong-anchor"),
            Ok(History {
                accepted: None,
                prepared: None
            })
        ));
    }

    #[test]
    fn late_finalize_is_idempotent_after_recovery_and_a_later_successor() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let original_store = db.open();
        let first = initialize(&original_store, &anchor, "log-late-finalize");
        let checkpoint_two = h(b"checkpoint-two");
        let receipt_two = h(b"receipt-two");
        let candidate_two = Record::build(
            2,
            "log-late-finalize",
            "policy-v1",
            checkpoint_two,
            1,
            Some(receipt_two),
            Some(first.digest),
        );

        // Simulate the original caller pausing after external CAS but before its
        // local finalize transaction.
        assert!(matches!(
            original_store.transition_inner(
                "log-late-finalize",
                first.generation,
                first.digest,
                None,
                checkpoint_two,
                1,
                Some(receipt_two),
                &anchor,
                Some(FaultPoint::AfterExternalAnchorAdvance),
            ),
            Err(WitnessError::InjectedCrash(FaultPoint::AfterExternalAnchorAdvance))
        ));

        // Another store instance recovers and finalizes the same prepared
        // candidate, then successfully advances to generation 3.
        let recovery_store = db.open();
        let recovered_two = recovery_store
            .recover("log-late-finalize", &anchor)
            .expect("recover prepared candidate")
            .expect("accepted generation two");
        assert_eq!(recovered_two, candidate_two);
        let third = recovery_store
            .advance(
                "log-late-finalize",
                recovered_two.generation,
                recovered_two.digest,
                h(b"checkpoint-three"),
                2,
                Some(h(b"receipt-three")),
                &anchor,
            )
            .expect("advance after recovery");
        assert_eq!(third.generation, 3);

        // The original caller's late finalize sees a newer head, but its exact
        // generation-two record is already accepted. It must not report a
        // false stale-predecessor failure or move the head backwards.
        original_store
            .finalize(&candidate_two, first.generation, first.digest)
            .expect("late finalize of accepted historical candidate is idempotent");
        assert_eq!(
            original_store
                .recover("log-late-finalize", &anchor)
                .expect("head remains valid"),
            Some(third)
        );
    }

    #[test]
    fn finalize_rejects_tampered_current_head_record_on_same_generation_retry() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let log_id = "log-same-head-tampered-row";
        let first = initialize(&store, &anchor, log_id);

        // The metadata digest and record-digest column remain unchanged, but
        // a canonical record field is altered after acceptance.
        let conn = store
            .open_connection()
            .expect("open connection for corruption injection");
        conn.execute(
            "UPDATE witness_records SET anchor_digest=?1 WHERE log_id=?2 AND generation=1",
            params![h(b"tampered-current-head-anchor").as_slice(), log_id],
        )
        .expect("tamper accepted current-head field");
        drop(conn);

        assert!(matches!(
            store.finalize(&first, 0, ZERO_DIGEST),
            Err(WitnessError::CorruptStore(
                "record chain, policy, or receipt history is invalid"
            ))
        ));
    }
    #[test]
    fn late_finalize_rejects_corrupt_current_head_metadata_pointer() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let original_store = db.open();
        let log_id = "log-late-finalize-corrupt-head";
        let first = initialize(&original_store, &anchor, log_id);
        let checkpoint_two = h(b"corrupt-head-checkpoint-two");
        let receipt_two = h(b"corrupt-head-receipt-two");
        let candidate_two = Record::build(
            2,
            log_id,
            "policy-v1",
            checkpoint_two,
            1,
            Some(receipt_two),
            Some(first.digest),
        );

        assert!(matches!(
            original_store.transition_inner(
                log_id,
                first.generation,
                first.digest,
                None,
                checkpoint_two,
                1,
                Some(receipt_two),
                &anchor,
                Some(FaultPoint::AfterExternalAnchorAdvance),
            ),
            Err(WitnessError::InjectedCrash(FaultPoint::AfterExternalAnchorAdvance))
        ));

        let recovery_store = db.open();
        let recovered_two = recovery_store
            .recover(log_id, &anchor)
            .expect("recover prepared candidate")
            .expect("accepted generation two");
        assert_eq!(recovered_two, candidate_two);
        recovery_store
            .advance(
                log_id,
                recovered_two.generation,
                recovered_two.digest,
                h(b"corrupt-head-checkpoint-three"),
                2,
                Some(h(b"corrupt-head-receipt-three")),
                &anchor,
            )
            .expect("advance after recovery");

        // Keep the pointer well-formed (32 bytes) but make it disagree with
        // the accepted record at the current head generation.
        let conn = original_store
            .open_connection()
            .expect("open connection for corruption injection");
        conn.execute(
            "UPDATE witness_meta SET record_digest=?1 WHERE log_id=?2",
            params![h(b"not-the-current-head").as_slice(), log_id],
        )
        .expect("corrupt metadata pointer");
        drop(conn);

        assert!(matches!(
            original_store.finalize(&candidate_two, first.generation, first.digest),
            Err(WitnessError::CorruptStore(
                "metadata record digest does not match current accepted head"
            ))
        ));
    }
    #[test]
    fn late_finalize_rejects_tampered_current_head_record() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let original_store = db.open();
        let log_id = "log-late-finalize-tampered-row";
        let first = initialize(&original_store, &anchor, log_id);
        let checkpoint_two = h(b"tampered-row-checkpoint-two");
        let receipt_two = h(b"tampered-row-receipt-two");
        let candidate_two = Record::build(
            2,
            log_id,
            "policy-v1",
            checkpoint_two,
            1,
            Some(receipt_two),
            Some(first.digest),
        );

        assert!(matches!(
            original_store.transition_inner(
                log_id,
                first.generation,
                first.digest,
                None,
                checkpoint_two,
                1,
                Some(receipt_two),
                &anchor,
                Some(FaultPoint::AfterExternalAnchorAdvance),
            ),
            Err(WitnessError::InjectedCrash(FaultPoint::AfterExternalAnchorAdvance))
        ));

        let recovery_store = db.open();
        let recovered_two = recovery_store
            .recover(log_id, &anchor)
            .expect("recover prepared candidate")
            .expect("accepted generation two");
        assert_eq!(recovered_two, candidate_two);
        recovery_store
            .advance(
                log_id,
                recovered_two.generation,
                recovered_two.digest,
                h(b"tampered-row-checkpoint-three"),
                2,
                Some(h(b"tampered-row-receipt-three")),
                &anchor,
            )
            .expect("advance after recovery");

        // Preserve the stored digest column and metadata pointer, but mutate
        // a committed field so the current row no longer hashes to that digest.
        let conn = original_store
            .open_connection()
            .expect("open connection for corruption injection");
        conn.execute(
            "UPDATE witness_records SET anchor_digest=?1 WHERE log_id=?2 AND generation=3",
            params![h(b"tampered-current-head-anchor").as_slice(), log_id],
        )
        .expect("tamper accepted current-head field");
        drop(conn);

        assert!(matches!(
            original_store.finalize(&candidate_two, first.generation, first.digest),
            Err(WitnessError::CorruptStore(
                "record chain, policy, or receipt history is invalid"
            ))
        ));
    }

    #[test]
    fn anchor_unavailable_after_prepare_leaves_candidate_unaccepted_then_retry_recovers() {
        let db = TempDb::new();
        let anchor = OneShotUnavailableAnchor::new();
        anchor.provision("log-anchor-outage");
        let store = db.open();
        let first = store
            .initialize(
                "log-anchor-outage",
                "policy-v1",
                h(b"genesis-checkpoint"),
                0,
                None,
                &anchor,
            )
            .expect("initialize before outage");

        // Advance uses current() once for recovery, once before preparation,
        // and once after the durable prepare. Fail the third read so the
        // candidate is persisted but the external anchor is never advanced.
        anchor.fail_on_nth_next_current(3);
        assert!(matches!(
            store.advance(
                "log-anchor-outage",
                first.generation,
                first.digest,
                h(b"checkpoint-two"),
                1,
                Some(h(b"receipt-one")),
                &anchor,
            ),
            Err(WitnessError::Anchor(AnchorError::Unavailable))
        ));

        assert_eq!(
            store.load_history("log-anchor-outage").expect("local history"),
            History {
                accepted: Some(first.clone()),
                prepared: Some(Record::build(
                    2,
                    "log-anchor-outage",
                    "policy-v1",
                    h(b"checkpoint-two"),
                    1,
                    Some(h(b"receipt-one")),
                    Some(first.digest),
                )),
            }
        );
        assert_eq!(
            store.recover("log-anchor-outage", &anchor).expect("recover old accepted head"),
            Some(first.clone())
        );

        let second = store
            .advance(
                "log-anchor-outage",
                first.generation,
                first.digest,
                h(b"checkpoint-two"),
                1,
                Some(h(b"receipt-one")),
                &anchor,
            )
            .expect("identical retry after anchor returns");
        assert_eq!(second.generation, 2);
        assert_eq!(
            store.recover("log-anchor-outage", &anchor).expect("recover accepted retry"),
            Some(second)
        );
    }

    #[test]
    fn prepared_candidate_is_reused_after_restart_when_anchor_is_unchanged() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-retry");
        let checkpoint = h(b"checkpoint-two");
        let receipt = h(b"receipt-one");
        assert!(matches!(
            store.transition_inner(
                "log-retry",
                first.generation,
                first.digest,
                None,
                checkpoint,
                1,
                Some(receipt),
                &anchor,
                Some(FaultPoint::AfterPrepare),
            ),
            Err(WitnessError::InjectedCrash(FaultPoint::AfterPrepare))
        ));
        drop(store);

        let reopened = db.open();
        assert_eq!(
            reopened.recover("log-retry", &anchor).expect("prior state"),
            Some(first.clone())
        );
        let second = reopened.advance(
            "log-retry",
            first.generation,
            first.digest,
            checkpoint,
            1,
            Some(receipt),
            &anchor,
        ).expect("retry prepared candidate");
        assert_eq!(second.generation, 2);
    }

    #[test]
    fn competing_prepared_candidates_preserve_fork_evidence() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-fork");

        let result_one = store.transition_inner(
            "log-fork",
            first.generation,
            first.digest,
            None,
            h(b"candidate-one"),
            1,
            Some(h(b"receipt-one")),
            &anchor,
            Some(FaultPoint::AfterPrepare),
        );
        assert!(matches!(result_one, Err(WitnessError::InjectedCrash(_))));

        assert!(matches!(
            store.advance(
                "log-fork",
                first.generation,
                first.digest,
                h(b"candidate-two"),
                1,
                Some(h(b"receipt-two")),
                &anchor,
            ),
            Err(WitnessError::PreparedCandidateConflict)
        ));
        assert_eq!(store.integrity_check().expect("SQLite integrity"), "ok");
        assert!(matches!(
            store.load_history("log-fork"),
            Ok(History { accepted: Some(_), prepared: Some(_) })
        ));
    }

    #[test]
    fn two_store_instances_cannot_both_advance_one_predecessor() {
        let db = TempDb::new();
        let anchor = Arc::new(MemoryAnchor::default());
        let store = db.open();
        let first = initialize(&store, anchor.as_ref(), "log-race");
        drop(store);

        let one = db.open();
        let two = db.open();
        let a1 = Arc::clone(&anchor);
        let a2 = Arc::clone(&anchor);
        let predecessor = first.clone();
        let handle_one = thread::spawn(move || {
            one.advance(
                "log-race",
                predecessor.generation,
                predecessor.digest,
                h(b"race-one"),
                1,
                Some(h(b"receipt-one")),
                a1.as_ref(),
            )
        });
        let predecessor = first.clone();
        let handle_two = thread::spawn(move || {
            two.advance(
                "log-race",
                predecessor.generation,
                predecessor.digest,
                h(b"race-two"),
                1,
                Some(h(b"receipt-two")),
                a2.as_ref(),
            )
        });

        let outcomes = [
            handle_one.join().expect("thread one"),
            handle_two.join().expect("thread two"),
        ];
        assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
        assert_eq!(storeless_recover(&db, anchor.as_ref(), "log-race").generation, 2);
    }

    #[test]
    fn anchor_already_ahead_before_prepare_is_not_recorded_as_fork() {
        let db = TempDb::new();
        let store = db.open();
        let log_id = "log-anchor-ahead-before-prepare";
        let initial_anchor = MemoryAnchor::default();
        let first = store
            .initialize(
                log_id,
                "policy-v1",
                h(b"checkpoint-one"),
                0,
                None,
                &initial_anchor,
            )
            .expect("initialize first record");
        let advanced = AnchorState {
            log_id: log_id.to_owned(),
            generation: 3,
            record_digest: h(b"later-external-generation-three"),
        };
        // Recovery reads once; the pre-prepare observation is the second read.
        let racing = AdvanceDuringReadAnchor::new(
            log_id,
            AnchorState::from_record(&first),
            2,
            advanced,
        );
        assert!(matches!(
            store.advance(
                log_id,
                first.generation,
                first.digest,
                h(b"candidate-generation-two"),
                1,
                Some(h(b"receipt-one")),
                &racing,
            ),
            Err(WitnessError::RollbackDetected)
        ));

        let history = store.load_history(log_id).expect("history remains valid");
        assert_eq!(history.accepted.as_ref(), Some(&first));
        assert!(history.prepared.is_none(), "pre-prepare rejection writes no candidate");
        let conn = store.open_connection().expect("open fork evidence query");
        let fork_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM witness_fork_evidence WHERE log_id=?1",
                params![log_id],
                |row| row.get(0),
            )
            .expect("count fork evidence");
        assert_eq!(fork_count, 0, "anchor-ahead state is not same-generation fork evidence");
    }

    #[test]
    fn anchor_that_advances_past_candidate_is_not_recorded_as_same_generation_fork() {
        let db = TempDb::new();
        let store = db.open();
        let log_id = "log-anchor-ahead-during-transition";
        let initial_anchor = MemoryAnchor::default();
        let first = store.initialize(
            log_id,
            "policy-v1",
            h(b"checkpoint-one"),
            0,
            None,
            &initial_anchor,
        ).expect("initialize first record");

        // transition_inner reads anchor.current() three times: recovery,
        // pre-prepare comparison, and post-prepare comparison. Simulate another
        // actor moving the independent anchor beyond our candidate at read 3.
        let advanced = AnchorState {
            log_id: log_id.to_owned(),
            generation: 3,
            record_digest: h(b"later-external-generation-three"),
        };
        let racing = AdvanceDuringReadAnchor::new(
            log_id,
            AnchorState::from_record(&first),
            3,
            advanced,
        );
        assert!(matches!(
            store.advance(
                log_id,
                first.generation,
                first.digest,
                h(b"candidate-generation-two"),
                1,
                Some(h(b"receipt-one")),
                &racing,
            ),
            Err(WitnessError::RollbackDetected)
        ));

        let history = store.load_history(log_id).expect("history remains inspectable");
        assert_eq!(history.accepted.as_ref(), Some(&first));
        assert_eq!(history.prepared.as_ref().map(|record| record.generation), Some(2));
        let conn = store.open_connection().expect("open fork evidence query");
        let fork_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM witness_fork_evidence WHERE log_id=?1",
            params![log_id],
            |row| row.get(0),
        ).expect("count fork evidence");
        assert_eq!(fork_count, 0, "cross-generation anchor state is not fork evidence");
    }

    #[test]
    fn external_anchor_race_fails_closed_and_fork_hash_is_log_scoped() {
        let db = TempDb::new();
        let anchor = RacingAnchor::new("log-external-race");
        let store = db.open();
        let first = store
            .initialize(
                "log-external-race",
                "policy-v1",
                h(b"genesis"),
                0,
                None,
                &anchor,
            )
            .expect("initialize before injecting race");

        anchor.arm_race();
        assert!(matches!(
            store.advance(
                "log-external-race",
                first.generation,
                first.digest,
                h(b"candidate-local"),
                1,
                Some(h(b"receipt-local")),
                &anchor,
            ),
            Err(WitnessError::ExternalAnchorMismatch)
        ));
        assert!(matches!(
            store.recover("log-external-race", &anchor),
            Err(WitnessError::RollbackDetected)
        ));
        assert!(matches!(
            store.load_history("log-external-race"),
            Ok(History { accepted: Some(_), prepared: Some(_) })
        ));

        let conn = store.open_connection().expect("open database for tamper test");
        conn.execute(
            "UPDATE witness_fork_evidence SET log_id=?1 WHERE log_id=?2",
            params!["replayed-as-another-log", "log-external-race"],
        )
        .expect("simulate cross-log evidence replay");
        assert!(matches!(
            store.load_history("replayed-as-another-log"),
            Err(WitnessError::CorruptForkEvidence)
        ));
    }

    #[test]
    fn fork_evidence_tail_truncation_is_detected() {
        let db = TempDb::new();
        let store = db.open();
        let log_id = "log-fork-tail-truncated";
        let first = h(b"fork-tail-first-record");
        let conflicting = h(b"fork-tail-conflicting-record");
        store
            .record_fork(log_id, 2, first, conflicting)
            .expect("append first fork evidence");
        store
            .record_fork(log_id, 3, h(b"fork-tail-second-first"), h(b"fork-tail-second-conflict"))
            .expect("append second fork evidence");
        {
            let conn = store.open_connection().expect("open fork store");
            assert_eq!(validate_fork_history(&conn, log_id), Ok(()));
            conn.execute(
                "DELETE FROM witness_fork_evidence WHERE log_id=?1 AND id=(
                    "SELECT MAX(id) FROM witness_fork_evidence WHERE log_id=?1)",
                params![log_id],
            )
            .expect("truncate last evidence row");
        }
        let conn = store.open_connection().expect("reopen fork store");
        assert!(matches!(validate_fork_history(&conn, log_id), Err(WitnessError::CorruptForkEvidence)));
    }

    #[test]
    fn fork_evidence_tail_pointer_tampering_is_detected() {
        let db = TempDb::new();
        let store = db.open();
        let log_id = "log-fork-tail-pointer-tampered";
        store
            .record_fork(log_id, 2, h(b"fork-pointer-first"), h(b"fork-pointer-conflict"))
            .expect("append fork evidence");
        let conn = store.open_connection().expect("open fork store");
        conn.execute(
            "UPDATE witness_fork_meta SET tail_digest=?1 WHERE log_id=?2",
            params![h(b"wrong-tail-pointer").as_slice(), log_id],
        )
        .expect("tamper fork tail pointer");
        assert!(matches!(validate_fork_history(&conn, log_id), Err(WitnessError::CorruptForkEvidence)));
    }

    #[test]
    fn restoring_old_local_history_is_detected_by_external_anchor() {
        let db = TempDb::new();
        let anchor = MemoryAnchor::default();
        let store = db.open();
        let first = initialize(&store, &anchor, "log-snapshot-rollback");
        store
            .advance(
                "log-snapshot-rollback",
                first.generation,
                first.digest,
                h(b"checkpoint-two"),
                1,
                Some(h(b"receipt-one")),
                &anchor,
            )
            .expect("advance external and local history");

        let conn = store.open_connection().expect("open database to simulate snapshot restore");
        conn.execute(
            "DELETE FROM witness_records WHERE log_id=?1 AND generation=2",
            params!["log-snapshot-rollback"],
        )
        .expect("remove newer local history");
        conn.execute(
            "UPDATE witness_meta SET generation=1, record_digest=?1 WHERE log_id=?2",
            params![first.digest.as_slice(), "log-snapshot-rollback"],
        )
        .expect("restore older metadata pointer");

        assert!(matches!(
            store.recover("log-snapshot-rollback", &anchor),
            Err(WitnessError::RollbackDetected)
        ));
    }

    fn storeless_recover(db: &TempDb, anchor: &MemoryAnchor, log_id: &str) -> Record {
        db.open()
            .recover(log_id, anchor)
            .expect("recover after raced advances")
            .expect("accepted state after raced advances")
    }
}
