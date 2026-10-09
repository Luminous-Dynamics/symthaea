//! Independently retained fork-evidence protocol primitives for CIV-014.
//!
//! This module specifies the event, frontier, receipt, and remote append contract.
//! It intentionally supplies no production witness backend and does not replace
//! the current adapter's local-only fork recording. Integrating these primitives
//! with a durable local pending journal and crash/recovery coordinator is a
//! separate step; until that exists, no external fork-audit guarantee is claimed.

use crate::Digest;
use super::{append_fork_evidence, SqliteWitnessStore, WitnessError};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;

const FORK_EVENT_SCHEMA_VERSION: u16 = 1;
const FORK_EVENT_DOMAIN: &[u8] = b"mycelix-civ014-fork-event-v1\0";
const FORK_RECEIPT_DOMAIN: &[u8] = b"mycelix-civ014-fork-receipt-v1\0";
const ZERO_DIGEST: Digest = [0; 32];

/// A monotonic frontier retained by a separately operated fork witness.
///
/// An empty frontier is valid only when explicitly provisioned. An unknown
/// log/epoch is unavailable, never implicitly initialized to genesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkFrontier {
    pub log_id: String,
    pub witness_epoch: u64,
    pub event_count: u64,
    pub tail_digest: Digest,
}

impl ForkFrontier {
    pub fn empty(log_id: impl Into<String>, witness_epoch: u64) -> Self {
        Self {
            log_id: log_id.into(),
            witness_epoch,
            event_count: 0,
            tail_digest: ZERO_DIGEST,
        }
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        if self.log_id.is_empty() {
            return Err(ForkWitnessError::InvalidInput("log ID is empty"));
        }
        if self.event_count == 0 && self.tail_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidFrontier(
                "empty frontier must use the zero tail digest",
            ));
        }
        Ok(())
    }

    fn after_event(&self, event: &ForkEvent) -> Result<Self, ForkWitnessError> {
        self.validate()?;
        event.validate_for(self)?;
        let event_count = self
            .event_count
            .checked_add(1)
            .ok_or(ForkWitnessError::FrontierOverflow)?;
        Ok(Self {
            log_id: self.log_id.clone(),
            witness_epoch: self.witness_epoch,
            event_count,
            tail_digest: event.event_digest,
        })
    }
}

/// Canonical fork event. The event digest is also the stable event identity.
///
/// The event records the accepted-head position observed when the conflict was
/// discovered, but it does not advance that accepted head. The previous fork
/// frontier count and tail are included in the hashed payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkEvent {
    pub schema_version: u16,
    pub log_id: String,
    pub witness_epoch: u64,
    pub previous_event_count: u64,
    pub previous_event_digest: Digest,
    pub accepted_generation: u64,
    pub accepted_head_digest: Digest,
    pub fork_generation: u64,
    pub first_record_digest: Digest,
    pub conflicting_record_digest: Digest,
    pub event_digest: Digest,
}

impl ForkEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        frontier: &ForkFrontier,
        accepted_generation: u64,
        accepted_head_digest: Digest,
        fork_generation: u64,
        first_record_digest: Digest,
        conflicting_record_digest: Digest,
    ) -> Result<Self, ForkWitnessError> {
        frontier.validate()?;
        if fork_generation == 0 {
            return Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            ));
        }
        if accepted_generation == 0 && accepted_head_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            ));
        }
        if fork_generation > accepted_generation.saturating_add(1) {
            return Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            ));
        }
        if first_record_digest == conflicting_record_digest {
            return Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            ));
        }
        let mut event = Self {
            schema_version: FORK_EVENT_SCHEMA_VERSION,
            log_id: frontier.log_id.clone(),
            witness_epoch: frontier.witness_epoch,
            previous_event_count: frontier.event_count,
            previous_event_digest: frontier.tail_digest,
            accepted_generation,
            accepted_head_digest,
            fork_generation,
            first_record_digest,
            conflicting_record_digest,
            event_digest: ZERO_DIGEST,
        };
        event.event_digest = event.calculate_digest()?;
        Ok(event)
    }

    /// Return the exact domain-separated canonical event encoding.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ForkWitnessError> {
        if self.schema_version != FORK_EVENT_SCHEMA_VERSION {
            return Err(ForkWitnessError::InvalidInput(
                "unsupported fork-event schema version",
            ));
        }
        if self.log_id.is_empty() {
            return Err(ForkWitnessError::InvalidInput("log ID is empty"));
        }
        if self.fork_generation == 0 {
            return Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            ));
        }
        if self.accepted_generation == 0 && self.accepted_head_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            ));
        }
        if self.fork_generation > self.accepted_generation.saturating_add(1) {
            return Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            ));
        }
        if self.first_record_digest == self.conflicting_record_digest {
            return Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            ));
        }
        if self.previous_event_count == 0 && self.previous_event_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "empty previous frontier must use zero digest",
            ));
        }

        let mut bytes = Vec::with_capacity(FORK_EVENT_DOMAIN.len() + self.log_id.len() + 202);
        bytes.extend_from_slice(FORK_EVENT_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        encode_field(&mut bytes, self.log_id.as_bytes());
        bytes.extend_from_slice(&self.witness_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.previous_event_count.to_be_bytes());
        bytes.extend_from_slice(&self.previous_event_digest);
        bytes.extend_from_slice(&self.accepted_generation.to_be_bytes());
        bytes.extend_from_slice(&self.accepted_head_digest);
        bytes.extend_from_slice(&self.fork_generation.to_be_bytes());
        bytes.extend_from_slice(&self.first_record_digest);
        bytes.extend_from_slice(&self.conflicting_record_digest);
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        if self.calculate_digest()? != self.event_digest {
            return Err(ForkWitnessError::InvalidEventDigest);
        }
        Ok(())
    }

    pub fn validate_for(&self, frontier: &ForkFrontier) -> Result<(), ForkWitnessError> {
        frontier.validate()?;
        self.validate()?;
        if self.log_id != frontier.log_id || self.witness_epoch != frontier.witness_epoch {
            return Err(ForkWitnessError::FrontierScopeMismatch);
        }
        if self.previous_event_count != frontier.event_count
            || self.previous_event_digest != frontier.tail_digest
        {
            return Err(ForkWitnessError::FrontierConflict);
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Result<Digest, ForkWitnessError> {
        Ok(Sha256::digest(self.canonical_bytes()?).into())
    }
}

/// Durable receipt returned by the independent fork witness.
///
/// The receipt commits to the exact event, prior frontier, and new frontier.
/// Callers must validate it against their pending local event before reporting
/// a completed append.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkAppendReceipt {
    pub event: ForkEvent,
    pub previous_frontier: ForkFrontier,
    pub frontier_after: ForkFrontier,
    pub receipt_digest: Digest,
}

impl ForkAppendReceipt {
    pub fn build(
        previous_frontier: ForkFrontier,
        event: ForkEvent,
    ) -> Result<Self, ForkWitnessError> {
        previous_frontier.validate()?;
        event.validate_for(&previous_frontier)?;
        let frontier_after = previous_frontier.after_event(&event)?;
        let mut receipt = Self {
            event,
            previous_frontier,
            frontier_after,
            receipt_digest: ZERO_DIGEST,
        };
        receipt.receipt_digest = receipt.calculate_digest()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        self.previous_frontier.validate()?;
        self.event.validate_for(&self.previous_frontier)?;
        let expected_after = self.previous_frontier.after_event(&self.event)?;
        if self.frontier_after != expected_after {
            return Err(ForkWitnessError::InvalidReceipt(
                "receipt frontier does not advance the prior frontier by one exact event",
            ));
        }
        if self.calculate_digest()? != self.receipt_digest {
            return Err(ForkWitnessError::InvalidReceiptDigest);
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Result<Digest, ForkWitnessError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FORK_RECEIPT_DOMAIN);
        encode_field(&mut bytes, self.previous_frontier.log_id.as_bytes());
        bytes.extend_from_slice(&self.previous_frontier.witness_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.previous_frontier.event_count.to_be_bytes());
        bytes.extend_from_slice(&self.previous_frontier.tail_digest);
        bytes.extend_from_slice(&self.event.event_digest);
        bytes.extend_from_slice(&self.frontier_after.event_count.to_be_bytes());
        bytes.extend_from_slice(&self.frontier_after.tail_digest);
        Ok(Sha256::digest(bytes).into())
    }
}

/// Separate capability from accepted-head CAS. Implementations must retain full
/// event payloads and receipts outside the local SQLite rollback domain.
///
/// append_event is linearizable and idempotent for the exact same event. If the
/// response is ambiguous, find_event resolves the stable event digest; read_event
/// lets recovery reconstruct a local journal whose fork rows were erased. A
/// frontier-only service cannot provide that reconstruction.
pub trait IndependentForkWitness: Send + Sync {
    fn current_frontier(
        &self,
        log_id: &str,
        witness_epoch: u64,
    ) -> Result<ForkFrontier, ForkWitnessError>;

    fn append_event(
        &self,
        expected_frontier: &ForkFrontier,
        event: &ForkEvent,
    ) -> Result<ForkAppendReceipt, ForkWitnessError>;

    fn find_event(
        &self,
        log_id: &str,
        witness_epoch: u64,
        event_digest: Digest,
    ) -> Result<Option<ForkAppendReceipt>, ForkWitnessError>;

    /// Sequence is one-based. A successful implementation retains the full
    /// event body, not just its digest, for local restoration after erasure.
    fn read_event(
        &self,
        log_id: &str,
        witness_epoch: u64,
        sequence: u64,
    ) -> Result<Option<ForkEvent>, ForkWitnessError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForkWitnessError {
    Unavailable,
    FrontierConflict,
    FrontierOverflow,
    FrontierScopeMismatch,
    EventIdentityConflict,
    InvalidEventDigest,
    InvalidFrontier(&'static str),
    InvalidInput(&'static str),
    InvalidReceipt(&'static str),
    InvalidReceiptDigest,
    LocalStore(String),
    LocalJournalCorrupt,
    RemoteHistoryIncomplete,
    AmbiguousAppend(String),
    Other(String),
}

impl fmt::Display for ForkWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "independent fork witness unavailable"),
            Self::FrontierConflict => write!(f, "independent fork-witness frontier conflict"),
            Self::FrontierOverflow => write!(f, "independent fork-witness frontier overflow"),
            Self::FrontierScopeMismatch => write!(f, "fork event is bound to another log or epoch"),
            Self::EventIdentityConflict => write!(f, "fork event identity conflicts with retained payload"),
            Self::InvalidEventDigest => write!(f, "fork-event digest does not match canonical payload"),
            Self::InvalidFrontier(message) => write!(f, "invalid fork-witness frontier: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid fork-witness input: {message}"),
            Self::InvalidReceipt(message) => write!(f, "invalid fork-witness receipt: {message}"),
            Self::InvalidReceiptDigest => write!(f, "fork-witness receipt digest mismatch"),
            Self::LocalStore(message) => write!(f, "local fork-witness journal error: {message}"),
            Self::LocalJournalCorrupt => write!(f, "local fork-witness journal is corrupt"),
            Self::RemoteHistoryIncomplete => write!(f, "remote fork-witness history is incomplete or inconsistent"),
            Self::AmbiguousAppend(message) => write!(f, "remote fork-witness append outcome is ambiguous: {message}"),
            Self::Other(message) => write!(f, "independent fork-witness error: {message}"),
        }
    }
}

impl std::error::Error for ForkWitnessError {}

fn encode_field(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredForkEvent {
    event: ForkEvent,
    state: i64,
    receipt: Option<ForkAppendReceipt>,
}

type RawForkEventRow = (
    String,
    i64,
    Vec<u8>,
    i64,
    i64,
    Vec<u8>,
    i64,
    Vec<u8>,
    i64,
    Vec<u8>,
    Vec<u8>,
    i64,
    Option<Vec<u8>>,
    Option<i64>,
    Option<Vec<u8>>,
);

fn local_store_error(error: impl fmt::Display) -> ForkWitnessError {
    ForkWitnessError::LocalStore(error.to_string())
}

fn digest_column(bytes: Vec<u8>) -> Result<Digest, ForkWitnessError> {
    bytes
        .try_into()
        .map_err(|_| ForkWitnessError::LocalJournalCorrupt)
}

fn sqlite_u64(value: u64, field: &'static str) -> Result<i64, ForkWitnessError> {
    i64::try_from(value).map_err(|_| {
        ForkWitnessError::InvalidInput(match field {
            "witness_epoch" => "witness epoch exceeds SQLite INTEGER range",
            "previous_event_count" => "fork frontier count exceeds SQLite INTEGER range",
            "accepted_generation" => "accepted generation exceeds SQLite INTEGER range",
            "fork_generation" => "fork generation exceeds SQLite INTEGER range",
            "receipt_event_count" => "receipt frontier count exceeds SQLite INTEGER range",
            _ => "integer field exceeds SQLite INTEGER range",
        })
    })
}

fn checked_u64(value: i64) -> Result<u64, ForkWitnessError> {
    u64::try_from(value).map_err(|_| ForkWitnessError::LocalJournalCorrupt)
}

fn decode_fork_event_row(raw: RawForkEventRow) -> Result<StoredForkEvent, ForkWitnessError> {
    let (
        log_id,
        epoch,
        event_digest,
        schema_version,
        previous_event_count,
        previous_event_digest,
        accepted_generation,
        accepted_head_digest,
        fork_generation,
        first_record_digest,
        conflicting_record_digest,
        state,
        receipt_digest,
        receipt_event_count,
        receipt_tail_digest,
    ) = raw;
    if epoch < 0 || schema_version < 0 || previous_event_count < 0
        || accepted_generation < 0 || fork_generation <= 0
    {
        return Err(ForkWitnessError::LocalJournalCorrupt);
    }
    let event = ForkEvent {
        schema_version: u16::try_from(schema_version)
            .map_err(|_| ForkWitnessError::LocalJournalCorrupt)?,
        log_id,
        witness_epoch: checked_u64(epoch)?,
        previous_event_count: checked_u64(previous_event_count)?,
        previous_event_digest: digest_column(previous_event_digest)?,
        accepted_generation: checked_u64(accepted_generation)?,
        accepted_head_digest: digest_column(accepted_head_digest)?,
        fork_generation: checked_u64(fork_generation)?,
        first_record_digest: digest_column(first_record_digest)?,
        conflicting_record_digest: digest_column(conflicting_record_digest)?,
        event_digest: digest_column(event_digest)?,
    };
    event.validate().map_err(|_| ForkWitnessError::LocalJournalCorrupt)?;
    match state {
        0 if receipt_digest.is_none()
            && receipt_event_count.is_none()
            && receipt_tail_digest.is_none() =>
        {
            Ok(StoredForkEvent { event, state, receipt: None })
        }
        1 => {
            let receipt_digest = digest_column(
                receipt_digest.ok_or(ForkWitnessError::LocalJournalCorrupt)?,
            )?;
            let receipt_event_count = checked_u64(
                receipt_event_count.ok_or(ForkWitnessError::LocalJournalCorrupt)?,
            )?;
            let receipt_tail_digest = digest_column(
                receipt_tail_digest.ok_or(ForkWitnessError::LocalJournalCorrupt)?,
            )?;
            let previous_frontier = ForkFrontier {
                log_id: event.log_id.clone(),
                witness_epoch: event.witness_epoch,
                event_count: event.previous_event_count,
                tail_digest: event.previous_event_digest,
            };
            let frontier_after = ForkFrontier {
                log_id: event.log_id.clone(),
                witness_epoch: event.witness_epoch,
                event_count: receipt_event_count,
                tail_digest: receipt_tail_digest,
            };
            let receipt = ForkAppendReceipt {
                event: event.clone(),
                previous_frontier,
                frontier_after,
                receipt_digest,
            };
            receipt.validate().map_err(|_| ForkWitnessError::LocalJournalCorrupt)?;
            Ok(StoredForkEvent { event, state, receipt: Some(receipt) })
        }
        _ => Err(ForkWitnessError::LocalJournalCorrupt),
    }
}

const SELECT_STORED_FORK_EVENT: &str =
    "SELECT log_id, witness_epoch, event_digest, schema_version,
            previous_event_count, previous_event_digest, accepted_generation,
            accepted_head_digest, fork_generation, first_record_digest,
            conflicting_record_digest, state, receipt_digest, receipt_event_count,
            receipt_tail_digest
     FROM witness_external_fork_events";

fn load_fork_event(
    conn: &Connection,
    log_id: &str,
    epoch: u64,
    event_digest: Digest,
) -> Result<Option<StoredForkEvent>, ForkWitnessError> {
    let epoch = sqlite_u64(epoch, "witness_epoch")?;
    let sql = format!(
        "{SELECT_STORED_FORK_EVENT}
         WHERE log_id=?1 AND witness_epoch=?2 AND event_digest=?3"
    );
    let raw: Option<RawForkEventRow> = conn
        .query_row(
            &sql,
            params![log_id, epoch, event_digest.as_slice()],
            |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?,
                    row.get(5)?, row.get(6)?, row.get(7)?, row.get(8)?, row.get(9)?,
                    row.get(10)?, row.get(11)?, row.get(12)?, row.get(13)?, row.get(14)?,
                ))
            },
        )
        .optional()
        .map_err(local_store_error)?;
    raw.map(decode_fork_event_row).transpose()
}

fn find_fork_event_by_conflict(
    conn: &Connection,
    log_id: &str,
    epoch: u64,
    accepted_generation: u64,
    accepted_head_digest: Digest,
    fork_generation: u64,
    first_record_digest: Digest,
    conflicting_record_digest: Digest,
) -> Result<Option<StoredForkEvent>, ForkWitnessError> {
    let (epoch, accepted_generation, fork_generation) = (
        sqlite_u64(epoch, "witness_epoch")?,
        sqlite_u64(accepted_generation, "accepted_generation")?,
        sqlite_u64(fork_generation, "fork_generation")?,
    );
    let sql = format!(
        "{SELECT_STORED_FORK_EVENT}
         WHERE log_id=?1 AND witness_epoch=?2 AND accepted_generation=?3
           AND accepted_head_digest=?4 AND fork_generation=?5
           AND first_record_digest=?6 AND conflicting_record_digest=?7
         ORDER BY previous_event_count ASC LIMIT 1"
    );
    let raw: Option<RawForkEventRow> = conn
        .query_row(
            &sql,
            params![
                log_id, epoch, accepted_generation, accepted_head_digest.as_slice(),
                fork_generation, first_record_digest.as_slice(),
                conflicting_record_digest.as_slice()
            ],
            |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?,
                    row.get(5)?, row.get(6)?, row.get(7)?, row.get(8)?, row.get(9)?,
                    row.get(10)?, row.get(11)?, row.get(12)?, row.get(13)?, row.get(14)?,
                ))
            },
        )
        .optional()
        .map_err(local_store_error)?;
    raw.map(decode_fork_event_row).transpose()
}

fn persist_pending_fork_event(
    store: &SqliteWitnessStore,
    frontier: &ForkFrontier,
    event: &ForkEvent,
) -> Result<(), ForkWitnessError> {
    event.validate_for(frontier)?;
    sqlite_u64(frontier.witness_epoch, "witness_epoch")?;
    sqlite_u64(frontier.event_count, "previous_event_count")?;
    sqlite_u64(event.accepted_generation, "accepted_generation")?;
    sqlite_u64(event.fork_generation, "fork_generation")?;

    let mut conn = store.open_connection().map_err(local_store_error)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(local_store_error)?;
    if let Some(existing) =
        load_fork_event(&tx, &event.log_id, event.witness_epoch, event.event_digest)?
    {
        if existing.event != *event {
            return Err(ForkWitnessError::EventIdentityConflict);
        }
        tx.commit().map_err(local_store_error)?;
        return Ok(());
    }

    tx.execute(
        "INSERT INTO witness_external_fork_events (
            log_id, witness_epoch, event_digest, schema_version,
            previous_event_count, previous_event_digest, accepted_generation,
            accepted_head_digest, fork_generation, first_record_digest,
            conflicting_record_digest, state, receipt_digest, receipt_event_count,
            receipt_tail_digest
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 0, NULL, NULL, NULL)",
        params![
            event.log_id,
            sqlite_u64(event.witness_epoch, "witness_epoch")?,
            event.event_digest.as_slice(),
            event.schema_version as i64,
            sqlite_u64(event.previous_event_count, "previous_event_count")?,
            event.previous_event_digest.as_slice(),
            sqlite_u64(event.accepted_generation, "accepted_generation")?,
            event.accepted_head_digest.as_slice(),
            sqlite_u64(event.fork_generation, "fork_generation")?,
            event.first_record_digest.as_slice(),
            event.conflicting_record_digest.as_slice(),
        ],
    )
    .map_err(local_store_error)?;
    tx.commit().map_err(local_store_error)?;
    Ok(())
}

fn local_fork_pair_exists(
    conn: &Connection,
    event: &ForkEvent,
) -> Result<bool, ForkWitnessError> {
    conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM witness_fork_evidence
            WHERE log_id=?1 AND generation=?2
              AND ((first_record_digest=?3 AND conflicting_record_digest=?4)
                OR (first_record_digest=?4 AND conflicting_record_digest=?3))
         )",
        params![
            event.log_id,
            sqlite_u64(event.fork_generation, "fork_generation")?,
            event.first_record_digest.as_slice(),
            event.conflicting_record_digest.as_slice()
        ],
        |row| row.get(0),
    )
    .map_err(local_store_error)
}

fn finalize_local_fork_receipt(
    store: &SqliteWitnessStore,
    receipt: &ForkAppendReceipt,
) -> Result<(), ForkWitnessError> {
    receipt.validate()?;
    let event = &receipt.event;
    let mut conn = store.open_connection().map_err(local_store_error)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(local_store_error)?;
    let stored = load_fork_event(&tx, &event.log_id, event.witness_epoch, event.event_digest)?
        .ok_or(ForkWitnessError::LocalJournalCorrupt)?;
    if stored.event != *event {
        return Err(ForkWitnessError::EventIdentityConflict);
    }
    if stored.receipt.as_ref().is_some_and(|existing| existing != receipt) {
        return Err(ForkWitnessError::InvalidReceipt(
            "receipt differs from the previously anchored receipt",
        ));
    }

    // Do not append the local evidence twice on retry. First validate the
    // existing chain; a fully erased (rows + tail metadata absent) local chain
    // can be rebuilt from the remote witness, while partial/corrupt local state
    // fails closed instead of being silently repaired.
    super::validate_fork_history(&tx, &event.log_id)
        .map_err(|_| ForkWitnessError::LocalJournalCorrupt)?;
    if !local_fork_pair_exists(&tx, event)? {
        append_fork_evidence(
            &tx,
            &event.log_id,
            event.fork_generation,
            event.first_record_digest,
            event.conflicting_record_digest,
        )
        .map_err(local_store_error)?;
    }
    if stored.state == 0 {
        tx.execute(
            "UPDATE witness_external_fork_events
             SET state=1, receipt_digest=?1, receipt_event_count=?2, receipt_tail_digest=?3
             WHERE log_id=?4 AND witness_epoch=?5 AND event_digest=?6 AND state=0",
            params![
                receipt.receipt_digest.as_slice(),
                sqlite_u64(receipt.frontier_after.event_count, "receipt_event_count")?,
                receipt.frontier_after.tail_digest.as_slice(),
                event.log_id,
                sqlite_u64(event.witness_epoch, "witness_epoch")?,
                event.event_digest.as_slice()
            ],
        )
        .map_err(local_store_error)?;
    }
    tx.commit().map_err(local_store_error)?;
    Ok(())
}

fn resolve_remote_append(
    witness: &dyn IndependentForkWitness,
    frontier: &ForkFrontier,
    event: &ForkEvent,
) -> Result<ForkAppendReceipt, ForkWitnessError> {
    let receipt = match witness.append_event(frontier, event) {
        Ok(receipt) => receipt,
        Err(append_error) => match witness.find_event(
            &event.log_id,
            event.witness_epoch,
            event.event_digest,
        ) {
            Ok(Some(receipt)) => receipt,
            Ok(None) => return Err(append_error),
            Err(read_error) => {
                return Err(ForkWitnessError::AmbiguousAppend(format!(
                    "append error: {append_error}; event lookup error: {read_error}"
                )));
            }
        },
    };
    receipt.validate()?;
    if receipt.event != *event || receipt.previous_frontier != *frontier {
        return Err(ForkWitnessError::InvalidReceipt(
            "remote receipt is not bound to the pending event and prior frontier",
        ));
    }
    Ok(receipt)
}

fn reconcile_stored_fork_event(
    store: &SqliteWitnessStore,
    stored: StoredForkEvent,
    witness: &dyn IndependentForkWitness,
) -> Result<ForkAppendReceipt, ForkWitnessError> {
    if let Some(receipt) = stored.receipt {
        finalize_local_fork_receipt(store, &receipt)?;
        return Ok(receipt);
    }
    let event = stored.event;
    match witness.find_event(&event.log_id, event.witness_epoch, event.event_digest)? {
        Some(receipt) => {
            receipt.validate()?;
            if receipt.event != event {
                return Err(ForkWitnessError::EventIdentityConflict);
            }
            finalize_local_fork_receipt(store, &receipt)?;
            Ok(receipt)
        }
        None => {
            let current = witness.current_frontier(&event.log_id, event.witness_epoch)?;
            let expected = ForkFrontier {
                log_id: event.log_id.clone(),
                witness_epoch: event.witness_epoch,
                event_count: event.previous_event_count,
                tail_digest: event.previous_event_digest,
            };
            if current != expected {
                if current.event_count > expected.event_count {
                    return Err(ForkWitnessError::RemoteHistoryIncomplete);
                }
                return Err(ForkWitnessError::FrontierConflict);
            }
            let receipt = resolve_remote_append(witness, &expected, &event)?;
            finalize_local_fork_receipt(store, &receipt)?;
            Ok(receipt)
        }
    }
}

impl SqliteWitnessStore {
    /// Persist a local pending event first, then append it to the independent
    /// witness, verify the exact receipt, and atomically finalize local fork
    /// evidence plus the journal state. This is an opt-in API; legacy transition
    /// calls still use the local-only fork-evidence path.
    pub fn record_fork_remotely(
        &self,
        accepted_generation: u64,
        accepted_head_digest: Digest,
        fork_generation: u64,
        first_record_digest: Digest,
        conflicting_record_digest: Digest,
        witness_epoch: u64,
        witness: &dyn IndependentForkWitness,
    ) -> Result<ForkAppendReceipt, ForkWitnessError> {
        let existing = {
            let conn = self.open_connection().map_err(local_store_error)?;
            find_fork_event_by_conflict(
                &conn,
                "",
                witness_epoch,
                accepted_generation,
                accepted_head_digest,
                fork_generation,
                first_record_digest,
                conflicting_record_digest,
            )?
        };
        // The log ID is derived from a provisioned remote frontier. A semantic
        // retry needs that same log ID, so read it from the caller's requested
        // conflict scope through the witness first when no local row is found.
        if let Some(stored) = existing {
            return reconcile_stored_fork_event(self, stored, witness);
        }

        // Probe the frontier before choosing the stable event identity.
        // Unknown log/epoch returns Unavailable; it is not silently provisioned.
        let frontier = witness.current_frontier("", witness_epoch)?;
        let event = ForkEvent::build(
            &frontier,
            accepted_generation,
            accepted_head_digest,
            fork_generation,
            first_record_digest,
            conflicting_record_digest,
        )?;
        persist_pending_fork_event(self, &frontier, &event)?;
        let receipt = resolve_remote_append(witness, &frontier, &event)?;
        finalize_local_fork_receipt(self, &receipt)?;
        Ok(receipt)
    }

    /// Reconcile every locally pending event in one log/epoch. Remote append
    /// ambiguity is resolved by stable event identity; if the witness frontier
    /// advanced but cannot return the pending payload/receipt, the method fails
    /// closed and leaves the local intent pending.
    pub fn reconcile_pending_fork_events(
        &self,
        log_id: &str,
        witness_epoch: u64,
        witness: &dyn IndependentForkWitness,
    ) -> Result<usize, ForkWitnessError> {
        let epoch = sqlite_u64(witness_epoch, "witness_epoch")?;
        let conn = self.open_connection().map_err(local_store_error)?;
        let digests: Vec<Vec<u8>> = {
            let mut statement = conn.prepare(
                "SELECT event_digest FROM witness_external_fork_events
                 WHERE log_id=?1 AND witness_epoch=?2 AND state=0
                 ORDER BY previous_event_count ASC, event_digest ASC",
            ).map_err(local_store_error)?;
            statement
                .query_map(params![log_id, epoch], |row| row.get(0))
                .map_err(local_store_error)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(local_store_error)?
        };
        drop(conn);
        let mut completed = 0usize;
        for raw_digest in digests {
            let event_digest = digest_column(raw_digest)?;
            let conn = self.open_connection().map_err(local_store_error)?;
            let stored = load_fork_event(&conn, log_id, witness_epoch, event_digest)?
                .ok_or(ForkWitnessError::LocalJournalCorrupt)?;
            drop(conn);
            if stored.state == 0 {
                reconcile_stored_fork_event(self, stored, witness)?;
                completed = completed.checked_add(1).ok_or(ForkWitnessError::FrontierOverflow)?;
            }
        }
        Ok(completed)
    }

    /// Rebuild a missing local fork log from the separately retained full event
    /// payloads. This operation refuses partial/corrupt local chains; only an
    /// entirely absent local fork chain is reconstructable by this version.
    pub fn restore_fork_history_from_witness(
        &self,
        log_id: &str,
        witness_epoch: u64,
        witness: &dyn IndependentForkWitness,
    ) -> Result<usize, ForkWitnessError> {
        let frontier = witness.current_frontier(log_id, witness_epoch)?;
        frontier.validate()?;
        if frontier.event_count > i64::MAX as u64 {
            return Err(ForkWitnessError::FrontierOverflow);
        }
        {
            let conn = self.open_connection().map_err(local_store_error)?;
            let (row_count, meta_present): (i64, i64) = conn.query_row(
                "SELECT (SELECT COUNT(*) FROM witness_fork_evidence WHERE log_id=?1),
                        (SELECT COUNT(*) FROM witness_fork_meta WHERE log_id=?1)",
                params![log_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).map_err(local_store_error)?;
            if row_count != 0 || meta_present != 0 {
                super::validate_fork_history(&conn, log_id)
                    .map_err(|_| ForkWitnessError::LocalJournalCorrupt)?;
            }
        }

        let mut expected = ForkFrontier::empty(log_id, witness_epoch);
        let mut restored = 0usize;
        for sequence in 1..=frontier.event_count {
            let event = witness
                .read_event(log_id, witness_epoch, sequence)?
                .ok_or(ForkWitnessError::RemoteHistoryIncomplete)?;
            event.validate_for(&expected)?;
            let receipt = witness
                .find_event(log_id, witness_epoch, event.event_digest)?
                .ok_or(ForkWitnessError::RemoteHistoryIncomplete)?;
            receipt.validate()?;
            let expected_receipt = ForkAppendReceipt::build(expected.clone(), event.clone())?;
            if receipt != expected_receipt {
                return Err(ForkWitnessError::RemoteHistoryIncomplete);
            }
            persist_pending_fork_event(self, &expected, &event)?;
            finalize_local_fork_receipt(self, &receipt)?;
            expected = receipt.frontier_after.clone();
            restored = restored.checked_add(1).ok_or(ForkWitnessError::FrontierOverflow)?;
        }
        if expected != frontier {
            return Err(ForkWitnessError::RemoteHistoryIncomplete);
        }
        Ok(restored)
    }
}

pub(crate) fn validate_external_fork_journal(
    conn: &Connection,
    log_id: &str,
) -> Result<(), &'static str> {
    let mut statement = conn
        .prepare(
            "SELECT event_digest FROM witness_external_fork_events
             WHERE log_id=?1 ORDER BY witness_epoch ASC, previous_event_count ASC",
        )
        .map_err(|_| "external fork journal query failed")?;
    let raw = statement
        .query_map(params![log_id], |row| row.get::<_, Vec<u8>>(0))
        .map_err(|_| "external fork journal query failed")?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "external fork journal row read failed")?;
    drop(statement);
    for bytes in raw {
        let digest = digest_column(bytes).map_err(|_| "external fork journal digest malformed")?;
        let stored = load_fork_event(conn, log_id, {
            let row: i64 = conn
                .query_row(
                    "SELECT witness_epoch FROM witness_external_fork_events
                     WHERE log_id=?1 AND event_digest=?2",
                    params![log_id, digest.as_slice()],
                    |row| row.get(0),
                )
                .map_err(|_| "external fork journal scope malformed")?;
            checked_u64(row).map_err(|_| "external fork journal scope malformed")?
        }, digest)
        .map_err(|_| "external fork journal row malformed")?
        .ok_or("external fork journal row missing")?;
        if stored.event.event_digest != digest {
            return Err("external fork journal identity mismatch");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryForkWitness {
        events: Mutex<HashMap<(String, u64), Vec<ForkAppendReceipt>>>,
    }

    impl MemoryForkWitness {
        fn provision(&self, log_id: &str, witness_epoch: u64) {
            self.events
                .lock()
                .expect("fork-witness mutex")
                .entry((log_id.to_owned(), witness_epoch))
                .or_default();
        }
    }

    impl IndependentForkWitness for MemoryForkWitness {
        fn current_frontier(
            &self,
            log_id: &str,
            witness_epoch: u64,
        ) -> Result<ForkFrontier, ForkWitnessError> {
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            Ok(receipts
                .last()
                .map(|receipt| receipt.frontier_after.clone())
                .unwrap_or_else(|| ForkFrontier::empty(log_id, witness_epoch)))
        }

        fn append_event(
            &self,
            expected_frontier: &ForkFrontier,
            event: &ForkEvent,
        ) -> Result<ForkAppendReceipt, ForkWitnessError> {
            expected_frontier.validate()?;
            event.validate_for(expected_frontier)?;
            let mut events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get_mut(&(expected_frontier.log_id.clone(), expected_frontier.witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;

            if let Some(existing) = receipts
                .iter()
                .find(|receipt| receipt.event.event_digest == event.event_digest)
            {
                if existing.event != *event {
                    return Err(ForkWitnessError::EventIdentityConflict);
                }
                existing.validate()?;
                return Ok(existing.clone());
            }

            let actual_frontier = receipts
                .last()
                .map(|receipt| receipt.frontier_after.clone())
                .unwrap_or_else(|| ForkFrontier::empty(
                    &expected_frontier.log_id,
                    expected_frontier.witness_epoch,
                ));
            if actual_frontier != *expected_frontier {
                return Err(ForkWitnessError::FrontierConflict);
            }
            let receipt = ForkAppendReceipt::build(expected_frontier.clone(), event.clone())?;
            receipt.validate()?;
            receipts.push(receipt.clone());
            Ok(receipt)
        }

        fn find_event(
            &self,
            log_id: &str,
            witness_epoch: u64,
            event_digest: Digest,
        ) -> Result<Option<ForkAppendReceipt>, ForkWitnessError> {
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            let found = receipts
                .iter()
                .find(|receipt| receipt.event.event_digest == event_digest)
                .cloned();
            if let Some(receipt) = found.as_ref() {
                receipt.validate()?;
            }
            Ok(found)
        }

        fn read_event(
            &self,
            log_id: &str,
            witness_epoch: u64,
            sequence: u64,
        ) -> Result<Option<ForkEvent>, ForkWitnessError> {
            if sequence == 0 {
                return Err(ForkWitnessError::InvalidInput(
                    "fork-event sequence is one-based",
                ));
            }
            let index = usize::try_from(sequence - 1)
                .map_err(|_| ForkWitnessError::InvalidInput("fork-event sequence is too large"))?;
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            Ok(receipts.get(index).map(|receipt| receipt.event.clone()))
        }
    }

    fn d(byte: u8) -> Digest {
        [byte; 32]
    }

    #[test]
    fn fork_event_and_receipt_are_canonical_and_bound_to_frontier() {
        let frontier = ForkFrontier::empty("log-a", 7);
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6))
            .expect("build canonical event");
        event.validate().expect("event digest verifies");
        let receipt = ForkAppendReceipt::build(frontier.clone(), event.clone())
            .expect("build receipt");
        receipt.validate().expect("receipt binds prior and next frontiers");
        assert_eq!(receipt.frontier_after.event_count, 1);
        assert_eq!(receipt.frontier_after.tail_digest, event.event_digest);
        assert_ne!(receipt.receipt_digest, event.event_digest);
        // Frozen independently from the Rust implementation: the fixture uses
        // log-a, epoch 7, empty prior frontier, accepted head (generation 4,
        // digest 0x04..), fork generation 4, and record digests 0x05../0x06...
        assert_eq!(
            event.event_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
            "85a478bb7489447b24b8874b8e0bcdb4ae143cd856174695e155ae4114c16dff",
        );
        assert_eq!(
            receipt.receipt_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
            "06df505452cc1ef6c45c717084da3f15c89f34958cee5ff0fdd4441bfb03b926",
        );
    }

    #[test]
    fn exact_remote_append_retry_returns_one_stable_receipt() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let before = witness.current_frontier("log-a", 7).expect("provisioned frontier");
        let event = ForkEvent::build(&before, 4, d(4), 4, d(5), d(6)).expect("build event");
        let first = witness.append_event(&before, &event).expect("append event");
        let retry = witness
            .append_event(&before, &event)
            .expect("exact retry returns existing receipt");
        assert_eq!(retry, first);
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            1,
            "idempotent retry cannot duplicate remote evidence",
        );
    }

    #[test]
    fn stale_frontier_cannot_append_a_different_event() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let empty = witness.current_frontier("log-a", 7).expect("empty frontier");
        let first = ForkEvent::build(&empty, 4, d(4), 4, d(5), d(6)).expect("first event");
        witness.append_event(&empty, &first).expect("first append");
        let competing = ForkEvent::build(&empty, 4, d(4), 4, d(7), d(8))
            .expect("competing event is canonical for stale frontier");
        assert_eq!(
            witness.append_event(&empty, &competing),
            Err(ForkWitnessError::FrontierConflict),
        );
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            1,
        );
    }

    #[test]
    fn invalid_event_digest_is_rejected_before_remote_append() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let frontier = witness.current_frontier("log-a", 7).expect("empty frontier");
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let mut tampered = event.clone();
        tampered.accepted_head_digest = d(9);
        assert_eq!(
            witness.append_event(&frontier, &tampered),
            Err(ForkWitnessError::InvalidEventDigest),
        );
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            0,
        );
    }

    #[test]
    fn remote_receipt_can_resolve_an_ambiguous_append_and_restore_payload() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let frontier = witness.current_frontier("log-a", 7).expect("empty frontier");
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let committed = witness.append_event(&frontier, &event).expect("remote commit");
        let found = witness
            .find_event("log-a", 7, event.event_digest)
            .expect("read back by stable identity")
            .expect("event exists remotely");
        assert_eq!(found, committed);
        let restored = witness
            .read_event("log-a", 7, 1)
            .expect("read retained payload")
            .expect("remote witness retains full body");
        assert_eq!(restored, event);
    }

    #[test]
    fn missing_remote_log_is_unavailable_not_implicit_genesis() {
        let witness = MemoryForkWitness::default();
        assert_eq!(
            witness.current_frontier("log-not-provisioned", 7),
            Err(ForkWitnessError::Unavailable),
        );
    }

    #[test]
    fn event_rejects_generation_zero_and_identical_competitors() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 0, ZERO_DIGEST, 0, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            )),
        );
        assert_eq!(
            ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(5)),
            Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            )),
        );
    }

    #[test]
    fn event_rejects_a_malformed_genesis_head_digest() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 0, d(9), 1, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            )),
        );
    }

    #[test]
    fn event_rejects_a_fork_generation_ahead_by_more_than_one() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 4, d(4), 6, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            )),
        );
    }

    #[test]
    fn receipt_tampering_is_detected() {
        let frontier = ForkFrontier::empty("log-a", 7);
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let mut receipt = ForkAppendReceipt::build(frontier, event).expect("build receipt");
        receipt.frontier_after.event_count = 2;
        assert_eq!(
            receipt.validate(),
            Err(ForkWitnessError::InvalidReceipt(
                "receipt frontier does not advance the prior frontier by one exact event",
            )),
        );
    }

    #[test]
    fn event_sequence_reads_are_one_based() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        assert_eq!(
            witness.read_event("log-a", 7, 0),
            Err(ForkWitnessError::InvalidInput("fork-event sequence is one-based")),
        );
    }
}
