// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable attempt evidence around the existing typed system broker boundaries.
//!
//! This crate does not reimplement `SystemdRecoveryBroker`. Instead it wraps the
//! broker's public `CheckpointStore` and `ServiceBackend` boundaries so every
//! call to the real restart backend is preceded by a durable `DispatchArmed`
//! record. If the process crashes immediately after the external effect, that
//! record already proves the attempt crossed the effect frontier and must be
//! treated conservatively as "may have occurred".

#![deny(unsafe_code)]

use std::error::Error as StdError;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, ReservationId};
use symthaea_authority::Digest32;
use symthaea_system_broker::{
    CheckpointStore, DispatchEvidence, HostId, RecoveryOutcome, RecoveryReceipt, ServiceBackend,
    ServiceObservation, ServiceUnit, VerificationResult,
};
use thiserror::Error;

pub const ATTEMPT_EVIDENCE_SCHEMA_VERSION: u16 = 1;
const ATTEMPT_CONTEXT_DOMAIN: &[u8] = b"symthaea.system-attempt.context.v1\0";
const ATTEMPT_RECORD_DOMAIN: &[u8] = b"symthaea.system-attempt.record.v1\0";
const EXECUTION_ID_DOMAIN: &[u8] = b"symthaea.system-attempt.execution-id.v1\0";
const RESERVATION_ID_DOMAIN: &[u8] = b"symthaea.system-attempt.reservation-id.v1\0";
const DIAGNOSTIC_DOMAIN: &[u8] = b"symthaea.system-attempt.diagnostic.v1\0";

/// Privacy-minimized identity of one exact broker attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidenceContext {
    pub schema_version: u16,
    pub execution_digest: Digest32,
    pub reservation_digest: Digest32,
    pub grant_digest: Digest32,
    pub plan_digest: Digest32,
    pub before_world_digest: Digest32,
    /// Optional commitment to an external authority artifact (for example the
    /// verified Xenia capability provenance used by #317).
    pub authority_evidence_digest: Option<Digest32>,
}

impl AttemptEvidenceContext {
    pub fn new(
        execution_id: &ExecutionId,
        reservation_id: &ReservationId,
        grant_digest: Digest32,
        plan_digest: Digest32,
        before_world_digest: Digest32,
        authority_evidence_digest: Option<Digest32>,
    ) -> Self {
        Self {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            execution_digest: digest_bytes(EXECUTION_ID_DOMAIN, execution_id.0.as_bytes()),
            reservation_digest: digest_bytes(RESERVATION_ID_DOMAIN, reservation_id.0.as_bytes()),
            grant_digest,
            plan_digest,
            before_world_digest,
            authority_evidence_digest,
        }
    }

    pub fn attempt_key(&self) -> Digest32 {
        digest_serialized(ATTEMPT_CONTEXT_DOMAIN, self)
    }
}

/// Durable evidence state for one attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptEvidenceState {
    /// Persisted immediately before calling the real restart backend. This means
    /// dispatch may occur after this record; absence of a later terminal record
    /// must therefore be treated as outcome-unknown.
    DispatchArmed,
    /// Backend evidence plus durable journal evidence says the effect was applied.
    Applied,
    /// Backend proved no external dispatch occurred and that fact reached the journal.
    ProvenNotDispatched,
    /// The effect may have occurred but cannot be proven either way.
    OutcomeUnknown,
    /// The enclosing broker completed its accounting + independent verification.
    RecoveryCompleted,
}

/// One append-only attempt evidence record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidenceRecord {
    pub schema_version: u16,
    pub context: AttemptEvidenceContext,
    pub sequence: u64,
    pub previous_evidence_digest: Option<Digest32>,
    /// Exact durable Agency Kernel checkpoint that existed at this evidence step.
    pub checkpoint_head: CheckpointHead,
    pub state: AttemptEvidenceState,
    pub diagnostic_digest: Option<Digest32>,
    pub after_world_digest: Option<Digest32>,
    pub recovery_outcome: Option<RecoveryOutcome>,
    pub verification: Option<VerificationResult>,
}

impl AttemptEvidenceRecord {
    pub fn digest(&self) -> Result<Digest32, AttemptEvidenceFormatError> {
        if self.schema_version != ATTEMPT_EVIDENCE_SCHEMA_VERSION
            || self.context.schema_version != ATTEMPT_EVIDENCE_SCHEMA_VERSION
        {
            return Err(AttemptEvidenceFormatError::UnsupportedSchema);
        }
        if self.sequence == 0 {
            if self.previous_evidence_digest.is_some() || self.state != AttemptEvidenceState::DispatchArmed {
                return Err(AttemptEvidenceFormatError::InvalidGenesis);
            }
        } else if self.previous_evidence_digest.is_none() {
            return Err(AttemptEvidenceFormatError::MissingPredecessor);
        }
        let encoded = bincode::serialize(self).map_err(|_| AttemptEvidenceFormatError::Encoding)?;
        Ok(digest_bytes(ATTEMPT_RECORD_DOMAIN, &encoded))
    }

    fn armed(context: AttemptEvidenceContext, checkpoint_head: CheckpointHead) -> Self {
        Self {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context,
            sequence: 0,
            previous_evidence_digest: None,
            checkpoint_head,
            state: AttemptEvidenceState::DispatchArmed,
            diagnostic_digest: None,
            after_world_digest: None,
            recovery_outcome: None,
            verification: None,
        }
    }

    fn successor(
        previous: AttemptEvidenceHead,
        context: AttemptEvidenceContext,
        checkpoint_head: CheckpointHead,
        state: AttemptEvidenceState,
        diagnostic_digest: Option<Digest32>,
    ) -> Result<Self, AttemptEvidenceFormatError> {
        Ok(Self {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context,
            sequence: previous
                .sequence
                .checked_add(1)
                .ok_or(AttemptEvidenceFormatError::SequenceOverflow)?,
            previous_evidence_digest: Some(previous.digest),
            checkpoint_head,
            state,
            diagnostic_digest,
            after_world_digest: None,
            recovery_outcome: None,
            verification: None,
        })
    }

    fn recovery_successor(
        previous: AttemptEvidenceHead,
        context: AttemptEvidenceContext,
        receipt: &RecoveryReceipt,
    ) -> Result<Self, AttemptEvidenceFormatError> {
        Ok(Self {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context,
            sequence: previous
                .sequence
                .checked_add(1)
                .ok_or(AttemptEvidenceFormatError::SequenceOverflow)?,
            previous_evidence_digest: Some(previous.digest),
            checkpoint_head: receipt.checkpoint_head,
            state: AttemptEvidenceState::RecoveryCompleted,
            diagnostic_digest: None,
            after_world_digest: receipt.after_world_digest,
            recovery_outcome: Some(receipt.outcome),
            verification: Some(receipt.verification),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidenceHead {
    pub sequence: u64,
    pub digest: Digest32,
}

/// Append-only durable evidence journal.
pub trait AttemptEvidenceJournal {
    type Error: StdError + 'static;

    fn append(
        &mut self,
        record: &AttemptEvidenceRecord,
    ) -> Result<AttemptEvidenceHead, Self::Error>;

    fn load_chain(
        &self,
        attempt_key: Digest32,
    ) -> Result<Vec<AttemptEvidenceRecord>, Self::Error>;
}

/// Shared handle retained by the orchestrator while the backend/store wrappers
/// are owned by `SystemdRecoveryBroker`.
pub struct AttemptEvidenceHandle<J> {
    context: AttemptEvidenceContext,
    journal: Arc<Mutex<J>>,
    last_head: Arc<Mutex<Option<AttemptEvidenceHead>>>,
}

impl<J> Clone for AttemptEvidenceHandle<J> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            journal: Arc::clone(&self.journal),
            last_head: Arc::clone(&self.last_head),
        }
    }
}

impl<J> AttemptEvidenceHandle<J>
where
    J: AttemptEvidenceJournal,
{
    pub fn context(&self) -> &AttemptEvidenceContext {
        &self.context
    }

    pub fn latest_head(&self) -> Result<Option<AttemptEvidenceHead>, AttemptEvidenceAccessError> {
        self.last_head
            .lock()
            .map(|guard| *guard)
            .map_err(|_| AttemptEvidenceAccessError::StatePoisoned)
    }

    pub fn load_chain(&self) -> Result<Vec<AttemptEvidenceRecord>, AttemptEvidenceAccessError> {
        self.journal
            .lock()
            .map_err(|_| AttemptEvidenceAccessError::JournalPoisoned)?
            .load_chain(self.context.attempt_key())
            .map_err(|error| AttemptEvidenceAccessError::Journal(diagnostic(&error)))
    }

    /// Append the successful broker-level accounting/verification result after
    /// `recover_once` returns a `RecoveryReceipt`.
    pub fn append_recovery_receipt(
        &self,
        receipt: &RecoveryReceipt,
    ) -> Result<AttemptEvidenceHead, AttemptEvidenceAccessError> {
        if receipt.grant_digest != self.context.grant_digest
            || receipt.plan_digest != self.context.plan_digest
            || receipt.before_world_digest != self.context.before_world_digest
        {
            return Err(AttemptEvidenceAccessError::ReceiptContextMismatch);
        }
        let previous = self
            .latest_head()?
            .ok_or(AttemptEvidenceAccessError::MissingDispatchEvidence)?;
        let record = AttemptEvidenceRecord::recovery_successor(
            previous,
            self.context.clone(),
            receipt,
        )
        .map_err(AttemptEvidenceAccessError::Format)?;
        let head = append_shared(&self.journal, &record)?;
        publish_last(&self.last_head, head)?;
        Ok(head)
    }
}

/// Wrap both public #305 boundaries around one shared attempt-evidence context.
pub fn instrument_attempt<B, S, J>(
    backend: B,
    checkpoint_store: S,
    journal: J,
    context: AttemptEvidenceContext,
) -> (
    EvidencedServiceBackend<B, J>,
    EvidencedCheckpointStore<S>,
    AttemptEvidenceHandle<J>,
)
where
    B: ServiceBackend,
    S: CheckpointStore,
    J: AttemptEvidenceJournal,
{
    let journal = Arc::new(Mutex::new(journal));
    let checkpoint_head = Arc::new(Mutex::new(None));
    let last_head = Arc::new(Mutex::new(None));

    let backend = EvidencedServiceBackend {
        inner: backend,
        context: context.clone(),
        journal: Arc::clone(&journal),
        checkpoint_head: Arc::clone(&checkpoint_head),
        last_head: Arc::clone(&last_head),
    };
    let store = EvidencedCheckpointStore {
        inner: checkpoint_store,
        checkpoint_head,
    };
    let handle = AttemptEvidenceHandle {
        context,
        journal,
        last_head,
    };
    (backend, store, handle)
}

/// Checkpoint wrapper that publishes the exact durable head to the effect wrapper.
pub struct EvidencedCheckpointStore<S> {
    inner: S,
    checkpoint_head: Arc<Mutex<Option<CheckpointHead>>>,
}

impl<S> CheckpointStore for EvidencedCheckpointStore<S>
where
    S: CheckpointStore,
    S::Error: 'static,
{
    type Error = EvidencedCheckpointStoreError<S::Error>;

    fn persist(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
        let head = self
            .inner
            .persist(checkpoint)
            .map_err(EvidencedCheckpointStoreError::Inner)?;
        let mut published = self
            .checkpoint_head
            .lock()
            .map_err(|_| EvidencedCheckpointStoreError::StatePoisoned)?;
        *published = Some(head);
        Ok(head)
    }
}

#[derive(Debug, Error)]
pub enum EvidencedCheckpointStoreError<E>
where
    E: StdError + 'static,
{
    #[error("inner checkpoint store failed: {0}")]
    Inner(#[source] E),
    #[error("checkpoint evidence publication state is poisoned")]
    StatePoisoned,
}

/// Service backend wrapper that durably arms evidence before the real effect.
pub struct EvidencedServiceBackend<B, J> {
    inner: B,
    context: AttemptEvidenceContext,
    journal: Arc<Mutex<J>>,
    checkpoint_head: Arc<Mutex<Option<CheckpointHead>>>,
    last_head: Arc<Mutex<Option<AttemptEvidenceHead>>>,
}

impl<B, J> ServiceBackend for EvidencedServiceBackend<B, J>
where
    B: ServiceBackend,
    J: AttemptEvidenceJournal,
{
    type Error = B::Error;

    fn observe(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error> {
        self.inner.observe(host, unit)
    }

    fn restart(
        &mut self,
        host: &HostId,
        unit: &ServiceUnit,
    ) -> Result<DispatchEvidence, Self::Error> {
        // If the broker's reservation checkpoint was not durably published to
        // this wrapper, do not call the real backend. This is proven no-dispatch.
        let reservation_head = match self
            .checkpoint_head
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            Some(head) => head,
            None => {
                return Ok(DispatchEvidence::NotDispatched {
                    diagnostic_digest: diagnostic_text("missing durable reservation checkpoint"),
                });
            }
        };

        // Durable write-ahead evidence: once this succeeds, a crash with no
        // terminal record means the external effect may have occurred.
        let armed = AttemptEvidenceRecord::armed(self.context.clone(), reservation_head);
        let armed_head = match append_shared(&self.journal, &armed) {
            Ok(head) => head,
            Err(error) => {
                return Ok(DispatchEvidence::NotDispatched {
                    diagnostic_digest: error.diagnostic_digest(),
                });
            }
        };
        if publish_last(&self.last_head, armed_head).is_err() {
            return Ok(DispatchEvidence::NotDispatched {
                diagnostic_digest: diagnostic_text("attempt evidence state publication failed"),
            });
        }

        let inner_result = self.inner.restart(host, unit);
        let (state, diagnostic_digest, outward) = match inner_result {
            Ok(DispatchEvidence::Applied) => (
                AttemptEvidenceState::Applied,
                None,
                DispatchEvidence::Applied,
            ),
            Ok(DispatchEvidence::NotDispatched { diagnostic_digest }) => (
                AttemptEvidenceState::ProvenNotDispatched,
                Some(diagnostic_digest),
                DispatchEvidence::NotDispatched { diagnostic_digest },
            ),
            Ok(DispatchEvidence::OutcomeUnknown { diagnostic_digest }) => (
                AttemptEvidenceState::OutcomeUnknown,
                Some(diagnostic_digest),
                DispatchEvidence::OutcomeUnknown { diagnostic_digest },
            ),
            Err(error) => {
                let digest = diagnostic(&error);
                (
                    AttemptEvidenceState::OutcomeUnknown,
                    Some(digest),
                    DispatchEvidence::OutcomeUnknown {
                        diagnostic_digest: digest,
                    },
                )
            }
        };

        let terminal = match AttemptEvidenceRecord::successor(
            armed_head,
            self.context.clone(),
            reservation_head,
            state,
            diagnostic_digest,
        ) {
            Ok(record) => record,
            Err(_) => {
                return Ok(DispatchEvidence::OutcomeUnknown {
                    diagnostic_digest: diagnostic_text("attempt evidence successor construction failed"),
                });
            }
        };

        match append_shared(&self.journal, &terminal) {
            Ok(head) => {
                if publish_last(&self.last_head, head).is_err() {
                    return Ok(DispatchEvidence::OutcomeUnknown {
                        diagnostic_digest: diagnostic_text("terminal evidence state publication failed"),
                    });
                }
                Ok(outward)
            }
            Err(error) => {
                // The armed record is already durable. Never return a definitive
                // success/not-dispatched classification if terminal evidence
                // could not be durably appended.
                Ok(DispatchEvidence::OutcomeUnknown {
                    diagnostic_digest: error.diagnostic_digest(),
                })
            }
        }
    }
}

fn append_shared<J>(
    journal: &Arc<Mutex<J>>,
    record: &AttemptEvidenceRecord,
) -> Result<AttemptEvidenceHead, AttemptEvidenceAccessError>
where
    J: AttemptEvidenceJournal,
{
    journal
        .lock()
        .map_err(|_| AttemptEvidenceAccessError::JournalPoisoned)?
        .append(record)
        .map_err(|error| AttemptEvidenceAccessError::Journal(diagnostic(&error)))
}

fn publish_last(
    last_head: &Arc<Mutex<Option<AttemptEvidenceHead>>>,
    head: AttemptEvidenceHead,
) -> Result<(), AttemptEvidenceAccessError> {
    let mut guard = last_head
        .lock()
        .map_err(|_| AttemptEvidenceAccessError::StatePoisoned)?;
    *guard = Some(head);
    Ok(())
}

#[derive(Debug, Error)]
pub enum AttemptEvidenceAccessError {
    #[error("attempt evidence journal lock is poisoned")]
    JournalPoisoned,
    #[error("attempt evidence state lock is poisoned")]
    StatePoisoned,
    #[error("attempt evidence journal failed; diagnostic commitment {0:?}")]
    Journal(Digest32),
    #[error("attempt evidence format invalid: {0}")]
    Format(AttemptEvidenceFormatError),
    #[error("no durable dispatch evidence exists for this attempt")]
    MissingDispatchEvidence,
    #[error("recovery receipt does not match the attempt evidence context")]
    ReceiptContextMismatch,
}

impl AttemptEvidenceAccessError {
    fn diagnostic_digest(&self) -> Digest32 {
        diagnostic_text(&self.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AttemptEvidenceFormatError {
    #[error("unsupported attempt evidence schema")]
    UnsupportedSchema,
    #[error("attempt evidence generation zero must be DispatchArmed with no predecessor")]
    InvalidGenesis,
    #[error("attempt evidence successor is missing its predecessor")]
    MissingPredecessor,
    #[error("attempt evidence sequence overflow")]
    SequenceOverflow,
    #[error("attempt evidence encoding failed")]
    Encoding,
}

// ---- SQLite append-only journal ------------------------------------------------

const SQLITE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS system_attempt_evidence (
    attempt_key BLOB NOT NULL CHECK (length(attempt_key) = 32),
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    digest BLOB NOT NULL CHECK (length(digest) = 32),
    record BLOB NOT NULL,
    PRIMARY KEY(attempt_key, sequence)
);
"#;

/// Concrete append-only SQLite journal for attempt evidence.
pub struct SqliteAttemptEvidenceJournal {
    connection: Connection,
}

impl SqliteAttemptEvidenceJournal {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteAttemptEvidenceError> {
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;\n\
             PRAGMA synchronous=FULL;\n\
             PRAGMA foreign_keys=ON;\n\
             PRAGMA busy_timeout=5000;",
        )?;
        connection.execute_batch(SQLITE_SCHEMA)?;
        Ok(Self { connection })
    }

    pub fn open_in_memory() -> Result<Self, SqliteAttemptEvidenceError> {
        let connection = Connection::open_in_memory()?;
        connection.execute_batch("PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;")?;
        connection.execute_batch(SQLITE_SCHEMA)?;
        Ok(Self { connection })
    }
}

impl AttemptEvidenceJournal for SqliteAttemptEvidenceJournal {
    type Error = SqliteAttemptEvidenceError;

    fn append(
        &mut self,
        record: &AttemptEvidenceRecord,
    ) -> Result<AttemptEvidenceHead, Self::Error> {
        let record_digest = record.digest().map_err(SqliteAttemptEvidenceError::Format)?;
        let attempt_key = record.context.attempt_key();
        let sequence = i64::try_from(record.sequence)
            .map_err(|_| SqliteAttemptEvidenceError::SequenceOutOfRange)?;
        let record_bytes =
            bincode::serialize(record).map_err(|_| SqliteAttemptEvidenceError::Encoding)?;

        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = latest_head_in_transaction(&transaction, attempt_key)?;
        let expected = if record.sequence == 0 {
            None
        } else {
            Some(AttemptEvidenceHead {
                sequence: record.sequence - 1,
                digest: record
                    .previous_evidence_digest
                    .ok_or(SqliteAttemptEvidenceError::MissingPredecessor)?,
            })
        };
        if current != expected {
            return Err(SqliteAttemptEvidenceError::Conflict { expected, actual: current });
        }

        transaction.execute(
            "INSERT INTO system_attempt_evidence(attempt_key, sequence, digest, record) VALUES(?1, ?2, ?3, ?4)",
            params![
                &attempt_key.0[..],
                sequence,
                &record_digest.0[..],
                record_bytes
            ],
        )?;
        transaction.commit()?;

        let head = AttemptEvidenceHead {
            sequence: record.sequence,
            digest: record_digest,
        };
        let durable = latest_head_in_connection(&self.connection, attempt_key)?
            .ok_or(SqliteAttemptEvidenceError::DurableReadbackMissing)?;
        if durable != head {
            return Err(SqliteAttemptEvidenceError::DurableHeadMismatch);
        }
        Ok(head)
    }

    fn load_chain(
        &self,
        attempt_key: Digest32,
    ) -> Result<Vec<AttemptEvidenceRecord>, Self::Error> {
        let mut statement = self.connection.prepare(
            "SELECT sequence, digest, record FROM system_attempt_evidence WHERE attempt_key = ?1 ORDER BY sequence ASC",
        )?;
        let mut rows = statement.query(params![&attempt_key.0[..]])?;
        let mut records = Vec::new();
        let mut previous: Option<AttemptEvidenceHead> = None;
        while let Some(row) = rows.next()? {
            let sequence: i64 = row.get(0)?;
            let digest: Vec<u8> = row.get(1)?;
            let bytes: Vec<u8> = row.get(2)?;
            let sequence = u64::try_from(sequence).map_err(|_| SqliteAttemptEvidenceError::CorruptRow)?;
            let digest: [u8; 32] = digest.try_into().map_err(|_| SqliteAttemptEvidenceError::CorruptRow)?;
            let record: AttemptEvidenceRecord =
                bincode::deserialize(&bytes).map_err(|_| SqliteAttemptEvidenceError::CorruptRecord)?;
            if record.context.attempt_key() != attempt_key || record.sequence != sequence {
                return Err(SqliteAttemptEvidenceError::CorruptRecord);
            }
            let computed = record.digest().map_err(SqliteAttemptEvidenceError::Format)?;
            let stored = AttemptEvidenceHead {
                sequence,
                digest: Digest32(digest),
            };
            if computed != stored.digest {
                return Err(SqliteAttemptEvidenceError::CorruptRecord);
            }
            match previous {
                None => {
                    if sequence != 0 || record.previous_evidence_digest.is_some() {
                        return Err(SqliteAttemptEvidenceError::CorruptRecord);
                    }
                }
                Some(head) => {
                    if sequence != head.sequence + 1
                        || record.previous_evidence_digest != Some(head.digest)
                    {
                        return Err(SqliteAttemptEvidenceError::CorruptRecord);
                    }
                }
            }
            previous = Some(stored);
            records.push(record);
        }
        Ok(records)
    }
}

fn latest_head_in_connection(
    connection: &Connection,
    attempt_key: Digest32,
) -> Result<Option<AttemptEvidenceHead>, SqliteAttemptEvidenceError> {
    connection
        .query_row(
            "SELECT sequence, digest FROM system_attempt_evidence WHERE attempt_key = ?1 ORDER BY sequence DESC LIMIT 1",
            params![&attempt_key.0[..]],
            decode_evidence_head,
        )
        .optional()?
        .map(validate_evidence_head)
        .transpose()
}

fn latest_head_in_transaction(
    transaction: &Transaction<'_>,
    attempt_key: Digest32,
) -> Result<Option<AttemptEvidenceHead>, SqliteAttemptEvidenceError> {
    transaction
        .query_row(
            "SELECT sequence, digest FROM system_attempt_evidence WHERE attempt_key = ?1 ORDER BY sequence DESC LIMIT 1",
            params![&attempt_key.0[..]],
            decode_evidence_head,
        )
        .optional()?
        .map(validate_evidence_head)
        .transpose()
}

fn decode_evidence_head(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, Vec<u8>)> {
    Ok((row.get(0)?, row.get(1)?))
}

fn validate_evidence_head(
    value: (i64, Vec<u8>),
) -> Result<AttemptEvidenceHead, SqliteAttemptEvidenceError> {
    let sequence = u64::try_from(value.0).map_err(|_| SqliteAttemptEvidenceError::CorruptRow)?;
    let digest: [u8; 32] = value.1.try_into().map_err(|_| SqliteAttemptEvidenceError::CorruptRow)?;
    Ok(AttemptEvidenceHead {
        sequence,
        digest: Digest32(digest),
    })
}

#[derive(Debug, Error)]
pub enum SqliteAttemptEvidenceError {
    #[error("SQLite attempt evidence operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("attempt evidence format invalid: {0}")]
    Format(AttemptEvidenceFormatError),
    #[error("attempt evidence encoding failed")]
    Encoding,
    #[error("attempt evidence sequence cannot be represented by SQLite INTEGER")]
    SequenceOutOfRange,
    #[error("attempt evidence successor is missing its predecessor")]
    MissingPredecessor,
    #[error("attempt evidence CAS conflict: expected {expected:?}, actual {actual:?}")]
    Conflict {
        expected: Option<AttemptEvidenceHead>,
        actual: Option<AttemptEvidenceHead>,
    },
    #[error("stored attempt evidence row is malformed")]
    CorruptRow,
    #[error("stored attempt evidence record is malformed or hash-inconsistent")]
    CorruptRecord,
    #[error("durable attempt evidence row disappeared after commit")]
    DurableReadbackMissing,
    #[error("durable attempt evidence head does not equal appended record")]
    DurableHeadMismatch,
}

fn digest_serialized<T: Serialize>(domain: &[u8], value: &T) -> Digest32 {
    let encoded = bincode::serialize(value).expect("security evidence serialization must be infallible");
    digest_bytes(domain, &encoded)
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    Digest32(*hasher.finalize().as_bytes())
}

fn diagnostic(error: &impl std::fmt::Display) -> Digest32 {
    diagnostic_text(&error.to_string())
}

fn diagnostic_text(value: &str) -> Digest32 {
    digest_bytes(DIAGNOSTIC_DOMAIN, value.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use symthaea_action_runtime::GrantAccount;
    use symthaea_authority::{AuthorityEpoch, CapabilityGrant, PrincipalId, RiskBudget};

    #[derive(Debug, Clone, Copy)]
    struct FakeError;
    impl fmt::Display for FakeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("fake error")
        }
    }
    impl StdError for FakeError {}

    struct FakeBackend {
        restart_calls: Arc<AtomicUsize>,
        dispatch: Result<DispatchEvidence, FakeError>,
        observations: VecDeque<ServiceObservation>,
    }

    impl ServiceBackend for FakeBackend {
        type Error = FakeError;
        fn observe(&mut self, _host: &HostId, _unit: &ServiceUnit) -> Result<ServiceObservation, Self::Error> {
            self.observations.pop_front().ok_or(FakeError)
        }
        fn restart(&mut self, _host: &HostId, _unit: &ServiceUnit) -> Result<DispatchEvidence, Self::Error> {
            self.restart_calls.fetch_add(1, Ordering::SeqCst);
            self.dispatch
        }
    }

    #[derive(Default)]
    struct FakeCheckpointStore;
    impl CheckpointStore for FakeCheckpointStore {
        type Error = FakeError;
        fn persist(&mut self, checkpoint: &GrantAccountCheckpoint) -> Result<CheckpointHead, Self::Error> {
            checkpoint.head().map_err(|_| FakeError)
        }
    }

    struct FailingJournal {
        records: Vec<AttemptEvidenceRecord>,
        fail_on_append: Option<usize>,
    }

    impl AttemptEvidenceJournal for FailingJournal {
        type Error = FakeError;
        fn append(&mut self, record: &AttemptEvidenceRecord) -> Result<AttemptEvidenceHead, Self::Error> {
            if self.fail_on_append == Some(self.records.len()) {
                return Err(FakeError);
            }
            let head = AttemptEvidenceHead {
                sequence: record.sequence,
                digest: record.digest().map_err(|_| FakeError)?,
            };
            self.records.push(record.clone());
            Ok(head)
        }
        fn load_chain(&self, _attempt_key: Digest32) -> Result<Vec<AttemptEvidenceRecord>, Self::Error> {
            Ok(self.records.clone())
        }
    }

    fn context() -> AttemptEvidenceContext {
        AttemptEvidenceContext::new(
            &ExecutionId("exec-1".into()),
            &ReservationId("reserve-1".into()),
            Digest32([1; 32]),
            Digest32([2; 32]),
            Digest32([3; 32]),
            Some(Digest32([4; 32])),
        )
    }

    fn checkpoint() -> GrantAccountCheckpoint {
        let mut grant = CapabilityGrant::new(
            "g",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            AuthorityEpoch(1),
        );
        grant.max_uses = 1;
        grant.risk_budget = RiskBudget {
            mutation_units: 1,
            ..RiskBudget::default()
        };
        GrantAccountCheckpoint::first(&grant, GrantAccount::new(&grant).snapshot()).unwrap()
    }

    fn host_unit() -> (HostId, ServiceUnit) {
        (
            HostId::parse("host-a").unwrap(),
            ServiceUnit::parse("postgresql.service").unwrap(),
        )
    }

    #[test]
    fn no_durable_armed_record_means_no_backend_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = FakeBackend {
            restart_calls: Arc::clone(&calls),
            dispatch: Ok(DispatchEvidence::Applied),
            observations: VecDeque::new(),
        };
        let journal = FailingJournal {
            records: Vec::new(),
            fail_on_append: Some(0),
        };
        let (mut backend, mut store, handle) =
            instrument_attempt(inner, FakeCheckpointStore, journal, context());
        store.persist(&checkpoint()).unwrap();
        let (host, unit) = host_unit();
        assert!(matches!(
            backend.restart(&host, &unit).unwrap(),
            DispatchEvidence::NotDispatched { .. }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(handle.load_chain().unwrap().is_empty());
    }

    #[test]
    fn terminal_journal_failure_keeps_armed_evidence_and_returns_unknown() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = FakeBackend {
            restart_calls: Arc::clone(&calls),
            dispatch: Ok(DispatchEvidence::Applied),
            observations: VecDeque::new(),
        };
        let journal = FailingJournal {
            records: Vec::new(),
            fail_on_append: Some(1),
        };
        let (mut backend, mut store, handle) =
            instrument_attempt(inner, FakeCheckpointStore, journal, context());
        store.persist(&checkpoint()).unwrap();
        let (host, unit) = host_unit();
        assert!(matches!(
            backend.restart(&host, &unit).unwrap(),
            DispatchEvidence::OutcomeUnknown { .. }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let records = handle.load_chain().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].state, AttemptEvidenceState::DispatchArmed);
    }

    #[test]
    fn successful_effect_has_armed_then_terminal_evidence() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = FakeBackend {
            restart_calls: Arc::clone(&calls),
            dispatch: Ok(DispatchEvidence::Applied),
            observations: VecDeque::new(),
        };
        let journal = SqliteAttemptEvidenceJournal::open_in_memory().unwrap();
        let (mut backend, mut store, handle) =
            instrument_attempt(inner, FakeCheckpointStore, journal, context());
        store.persist(&checkpoint()).unwrap();
        let (host, unit) = host_unit();
        assert_eq!(backend.restart(&host, &unit).unwrap(), DispatchEvidence::Applied);
        let records = handle.load_chain().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].state, AttemptEvidenceState::DispatchArmed);
        assert_eq!(records[1].state, AttemptEvidenceState::Applied);
        assert_eq!(records[1].previous_evidence_digest, Some(records[0].digest().unwrap()));
    }
}
