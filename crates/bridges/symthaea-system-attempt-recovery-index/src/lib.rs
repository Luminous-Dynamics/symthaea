// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only crash-recovery discovery for durable system-attempt evidence.
//!
//! This crate deliberately does **not** execute effects, mint capabilities, or
//! return capacity to a grant. Its job is narrower: after process restart it
//! discovers attempt chains from the SQLite evidence database, validates their
//! local hash/sequence/state-machine integrity, and classifies what the local
//! evidence can safely say.
//!
//! The central recovery rule is monotone:
//!
//! > A stranded `Reserved` execution normalizes to `OutcomeUnknown` unless an
//! > independently trusted proof establishes that dispatch did not occur.
//!
//! In particular, a local `ProvenNotDispatched` journal record is useful
//! evidence but is **not by itself authority to release capacity after a
//! crash**. The journal head is not yet anchored in an independent trust domain.

#![deny(unsafe_code)]

use std::path::Path;

use rusqlite::{params, Connection, OpenFlags};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_action_runtime::{ReservationId, ReservationState};
use symthaea_authority::Digest32;
use symthaea_system_attempt_evidence::{
    AttemptEvidenceContext, AttemptEvidenceFormatError, AttemptEvidenceHead,
    AttemptEvidenceRecord, AttemptEvidenceState,
};
use thiserror::Error;

const ATTEMPT_TABLE: &str = "system_attempt_evidence";
const RESERVATION_ID_DOMAIN_V1: &[u8] = b"symthaea.system-attempt.reservation-id.v1\0";

/// What a locally validated attempt chain is safe to contribute during restart.
///
/// These values are intentionally asymmetric: local evidence may keep or
/// strengthen an existing charge, but it never independently authorizes the
/// recovery layer to return authority to the grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalEvidenceDisposition {
    /// Dispatch may have happened. Keep the use/risk fully charged as unknown.
    ChargeAsOutcomeUnknown,
    /// Local evidence says the effect happened. This may support later
    /// reconciliation, but recovery remains non-executing.
    ChargeAsApplied,
    /// Local evidence claims no dispatch. Because the post-crash journal head is
    /// not independently anchored yet, this claim cannot by itself release the
    /// reservation.
    NonDispatchClaimNeedsIndependentTrust,
    /// The evidence chain contains the broker-level recovery completion record.
    /// A caller must still cross-check the authenticated Agency Kernel frontier;
    /// this status alone does not restore or mint authority.
    ClosedNeedsFrontierCrossCheck,
}

/// A fully revalidated attempt discovered from durable storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredAttempt {
    pub attempt_key: Digest32,
    pub context: AttemptEvidenceContext,
    pub head: AttemptEvidenceHead,
    pub last_state: AttemptEvidenceState,
    pub disposition: LocalEvidenceDisposition,
    pub record_count: usize,
    pub reservation_checkpoint_head: CheckpointHead,
    pub latest_checkpoint_head: CheckpointHead,
}

impl DiscoveredAttempt {
    pub fn is_closed(&self) -> bool {
        self.last_state == AttemptEvidenceState::RecoveryCompleted
    }

    pub fn matches_reservation_v1(&self, reservation_id: &ReservationId) -> bool {
        self.context.reservation_digest == reservation_id_digest_v1(reservation_id)
    }
}

/// Restart policy for the durable runtime reservation state itself.
///
/// No variant grants permission to dispatch. `Reserved` is conservatively
/// promoted to unknown because the crash boundary may have occurred after an
/// external effect and before the normal accounting transition became durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartReservationDisposition {
    KeepCommitted,
    KeepReleased,
    KeepOutcomeUnknown,
    NormalizeReservedToOutcomeUnknown,
}

pub fn restart_reservation_disposition(
    state: ReservationState,
) -> RestartReservationDisposition {
    match state {
        ReservationState::Committed => RestartReservationDisposition::KeepCommitted,
        ReservationState::Released => RestartReservationDisposition::KeepReleased,
        ReservationState::OutcomeUnknown => RestartReservationDisposition::KeepOutcomeUnknown,
        ReservationState::Reserved => {
            RestartReservationDisposition::NormalizeReservedToOutcomeUnknown
        }
    }
}

/// V1 compatibility commitment for the reservation identifier stored only as a
/// digest in attempt evidence.
///
/// This mirrors the schema-domain commitment used by
/// `symthaea-system-attempt-evidence` v1. A future evidence schema must add a new
/// helper rather than silently changing this function.
pub fn reservation_id_digest_v1(reservation_id: &ReservationId) -> Digest32 {
    digest_bytes(RESERVATION_ID_DOMAIN_V1, reservation_id.0.as_bytes())
}

/// Read-only SQLite recovery view.
///
/// Opening a separate read-only connection is intentional: crash recovery does
/// not reuse the live writer object or any in-process cached evidence head.
pub struct SqliteAttemptRecoveryIndex {
    connection: Connection,
}

impl SqliteAttemptRecoveryIndex {
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self, RecoveryIndexError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let connection = Connection::open_with_flags(path, flags)?;
        connection.execute_batch("PRAGMA query_only=ON; PRAGMA busy_timeout=5000;")?;
        Ok(Self { connection })
    }

    /// Discover every durable attempt and validate each complete local chain.
    ///
    /// Discovery does not trust the latest row alone. It enumerates keys, reloads
    /// every record in sequence order, recomputes every record digest, verifies
    /// predecessor linkage, requires a single immutable context, and enforces
    /// the v1 semantic state machine.
    pub fn scan_all(&self) -> Result<Vec<DiscoveredAttempt>, RecoveryIndexError> {
        let keys = self.attempt_keys()?;
        keys.into_iter().map(|key| self.load_attempt(key)).collect()
    }

    /// Return only attempts lacking a broker-level `RecoveryCompleted` record.
    pub fn scan_incomplete(&self) -> Result<Vec<DiscoveredAttempt>, RecoveryIndexError> {
        Ok(self
            .scan_all()?
            .into_iter()
            .filter(|attempt| !attempt.is_closed())
            .collect())
    }

    /// Locate the single incomplete attempt associated with a runtime reservation.
    /// Multiple matches are a containment condition rather than a tie-break.
    pub fn incomplete_for_reservation(
        &self,
        reservation_id: &ReservationId,
    ) -> Result<Option<DiscoveredAttempt>, RecoveryIndexError> {
        let target = reservation_id_digest_v1(reservation_id);
        let mut matches = self
            .scan_incomplete()?
            .into_iter()
            .filter(|attempt| attempt.context.reservation_digest == target);
        let first = matches.next();
        if matches.next().is_some() {
            return Err(RecoveryIndexError::AmbiguousReservationAttempt);
        }
        Ok(first)
    }

    fn attempt_keys(&self) -> Result<Vec<Digest32>, RecoveryIndexError> {
        let sql = format!(
            "SELECT DISTINCT attempt_key FROM {ATTEMPT_TABLE} ORDER BY hex(attempt_key) ASC"
        );
        let mut statement = self.connection.prepare(&sql)?;
        let mut rows = statement.query([])?;
        let mut keys = Vec::new();
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let key: [u8; 32] = bytes
                .try_into()
                .map_err(|_| RecoveryIndexError::MalformedAttemptKey)?;
            keys.push(Digest32(key));
        }
        Ok(keys)
    }

    fn load_attempt(&self, attempt_key: Digest32) -> Result<DiscoveredAttempt, RecoveryIndexError> {
        let sql = format!(
            "SELECT sequence, digest, record FROM {ATTEMPT_TABLE} \
             WHERE attempt_key = ?1 ORDER BY sequence ASC"
        );
        let mut statement = self.connection.prepare(&sql)?;
        let mut rows = statement.query(params![&attempt_key.0[..]])?;

        let mut previous: Option<(AttemptEvidenceHead, AttemptEvidenceState)> = None;
        let mut context: Option<AttemptEvidenceContext> = None;
        let mut first_checkpoint: Option<CheckpointHead> = None;
        let mut latest_checkpoint: Option<CheckpointHead> = None;
        let mut count = 0usize;

        while let Some(row) = rows.next()? {
            let stored_sequence: i64 = row.get(0)?;
            let stored_digest: Vec<u8> = row.get(1)?;
            let encoded: Vec<u8> = row.get(2)?;

            let sequence = u64::try_from(stored_sequence)
                .map_err(|_| RecoveryIndexError::MalformedSequence)?;
            let digest: [u8; 32] = stored_digest
                .try_into()
                .map_err(|_| RecoveryIndexError::MalformedDigest)?;
            let record: AttemptEvidenceRecord =
                bincode::deserialize(&encoded).map_err(|_| RecoveryIndexError::MalformedRecord)?;

            if record.sequence != sequence || record.context.attempt_key() != attempt_key {
                return Err(RecoveryIndexError::RecordIdentityMismatch);
            }
            let computed = record.digest()?;
            if computed != Digest32(digest) {
                return Err(RecoveryIndexError::RecordDigestMismatch);
            }

            match &context {
                None => {
                    context = Some(record.context.clone());
                    first_checkpoint = Some(record.checkpoint_head);
                }
                Some(expected) if expected != &record.context => {
                    return Err(RecoveryIndexError::ContextChangedWithinChain);
                }
                Some(_) => {}
            }

            validate_record_shape(&record)?;
            match previous {
                None => {
                    if record.sequence != 0 || record.previous_evidence_digest.is_some() {
                        return Err(RecoveryIndexError::InvalidGenesis);
                    }
                }
                Some((prior_head, prior_state)) => {
                    let expected_sequence = prior_head
                        .sequence
                        .checked_add(1)
                        .ok_or(RecoveryIndexError::SequenceOverflow)?;
                    if record.sequence != expected_sequence
                        || record.previous_evidence_digest != Some(prior_head.digest)
                    {
                        return Err(RecoveryIndexError::BrokenChain);
                    }
                    if !valid_state_transition(prior_state, record.state) {
                        return Err(RecoveryIndexError::InvalidStateTransition {
                            from: prior_state,
                            to: record.state,
                        });
                    }
                }
            }

            let head = AttemptEvidenceHead {
                sequence: record.sequence,
                digest: computed,
            };
            previous = Some((head, record.state));
            latest_checkpoint = Some(record.checkpoint_head);
            count = count
                .checked_add(1)
                .ok_or(RecoveryIndexError::SequenceOverflow)?;
        }

        let (head, last_state) = previous.ok_or(RecoveryIndexError::EmptyAttemptChain)?;
        let context = context.ok_or(RecoveryIndexError::EmptyAttemptChain)?;
        let reservation_checkpoint_head =
            first_checkpoint.ok_or(RecoveryIndexError::EmptyAttemptChain)?;
        let latest_checkpoint_head =
            latest_checkpoint.ok_or(RecoveryIndexError::EmptyAttemptChain)?;

        Ok(DiscoveredAttempt {
            attempt_key,
            context,
            head,
            last_state,
            disposition: disposition_for(last_state),
            record_count: count,
            reservation_checkpoint_head,
            latest_checkpoint_head,
        })
    }
}

fn disposition_for(state: AttemptEvidenceState) -> LocalEvidenceDisposition {
    match state {
        AttemptEvidenceState::DispatchArmed | AttemptEvidenceState::OutcomeUnknown => {
            LocalEvidenceDisposition::ChargeAsOutcomeUnknown
        }
        AttemptEvidenceState::Applied => LocalEvidenceDisposition::ChargeAsApplied,
        AttemptEvidenceState::ProvenNotDispatched => {
            LocalEvidenceDisposition::NonDispatchClaimNeedsIndependentTrust
        }
        AttemptEvidenceState::RecoveryCompleted => {
            LocalEvidenceDisposition::ClosedNeedsFrontierCrossCheck
        }
    }
}

fn validate_record_shape(record: &AttemptEvidenceRecord) -> Result<(), RecoveryIndexError> {
    match record.state {
        AttemptEvidenceState::DispatchArmed => {
            if record.sequence != 0
                || record.diagnostic_digest.is_some()
                || record.after_world_digest.is_some()
                || record.recovery_outcome.is_some()
                || record.verification.is_some()
            {
                return Err(RecoveryIndexError::InvalidRecordShape);
            }
        }
        AttemptEvidenceState::Applied
        | AttemptEvidenceState::ProvenNotDispatched
        | AttemptEvidenceState::OutcomeUnknown => {
            if record.sequence == 0
                || record.after_world_digest.is_some()
                || record.recovery_outcome.is_some()
                || record.verification.is_some()
            {
                return Err(RecoveryIndexError::InvalidRecordShape);
            }
        }
        AttemptEvidenceState::RecoveryCompleted => {
            if record.sequence == 0
                || record.recovery_outcome.is_none()
                || record.verification.is_none()
            {
                return Err(RecoveryIndexError::InvalidRecordShape);
            }
        }
    }
    Ok(())
}

fn valid_state_transition(from: AttemptEvidenceState, to: AttemptEvidenceState) -> bool {
    match from {
        AttemptEvidenceState::DispatchArmed => matches!(
            to,
            AttemptEvidenceState::Applied
                | AttemptEvidenceState::ProvenNotDispatched
                | AttemptEvidenceState::OutcomeUnknown
                | AttemptEvidenceState::RecoveryCompleted
        ),
        AttemptEvidenceState::Applied
        | AttemptEvidenceState::ProvenNotDispatched
        | AttemptEvidenceState::OutcomeUnknown => {
            to == AttemptEvidenceState::RecoveryCompleted
        }
        AttemptEvidenceState::RecoveryCompleted => false,
    }
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    Digest32(*hasher.finalize().as_bytes())
}

#[derive(Debug, Error)]
pub enum RecoveryIndexError {
    #[error("SQLite recovery-index operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("attempt evidence format invalid: {0}")]
    Format(#[from] AttemptEvidenceFormatError),
    #[error("stored attempt key is not a 32-byte digest")]
    MalformedAttemptKey,
    #[error("stored attempt sequence is invalid")]
    MalformedSequence,
    #[error("stored attempt digest is not a 32-byte digest")]
    MalformedDigest,
    #[error("stored attempt record cannot be decoded")]
    MalformedRecord,
    #[error("stored record sequence/context does not match its database identity")]
    RecordIdentityMismatch,
    #[error("stored record digest does not match the recomputed commitment")]
    RecordDigestMismatch,
    #[error("attempt evidence context changed within one hash chain")]
    ContextChangedWithinChain,
    #[error("attempt evidence chain is empty")]
    EmptyAttemptChain,
    #[error("attempt evidence genesis is malformed")]
    InvalidGenesis,
    #[error("attempt evidence predecessor sequence/digest link is broken")]
    BrokenChain,
    #[error("attempt evidence sequence overflow")]
    SequenceOverflow,
    #[error("attempt evidence record fields do not match its state")]
    InvalidRecordShape,
    #[error("invalid attempt evidence state transition {from:?} -> {to:?}")]
    InvalidStateTransition {
        from: AttemptEvidenceState,
        to: AttemptEvidenceState,
    },
    #[error("multiple incomplete attempts claim the same reservation")]
    AmbiguousReservationAttempt,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    use rusqlite::Connection;
    use symthaea_action_runtime::ExecutionId;
    use symthaea_system_attempt_evidence::{
        AttemptEvidenceJournal, SqliteAttemptEvidenceJournal, ATTEMPT_EVIDENCE_SCHEMA_VERSION,
    };

    fn temp_db(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-attempt-recovery-{label}-{}-{nonce}.sqlite",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let base = path.to_string_lossy();
        let _ = fs::remove_file(format!("{base}-wal"));
        let _ = fs::remove_file(format!("{base}-shm"));
    }

    fn context(reservation: &str) -> AttemptEvidenceContext {
        AttemptEvidenceContext::new(
            &ExecutionId(format!("exec-{reservation}")),
            &ReservationId(reservation.into()),
            Digest32([1; 32]),
            Digest32([2; 32]),
            Digest32([3; 32]),
            Some(Digest32([4; 32])),
        )
    }

    fn checkpoint(sequence: u64, byte: u8) -> CheckpointHead {
        CheckpointHead {
            sequence,
            digest: Digest32([byte; 32]),
        }
    }

    fn armed(context: AttemptEvidenceContext) -> AttemptEvidenceRecord {
        AttemptEvidenceRecord {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context,
            sequence: 0,
            previous_evidence_digest: None,
            checkpoint_head: checkpoint(1, 9),
            state: AttemptEvidenceState::DispatchArmed,
            diagnostic_digest: None,
            after_world_digest: None,
            recovery_outcome: None,
            verification: None,
        }
    }

    fn terminal(
        first: &AttemptEvidenceRecord,
        state: AttemptEvidenceState,
    ) -> AttemptEvidenceRecord {
        AttemptEvidenceRecord {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context: first.context.clone(),
            sequence: 1,
            previous_evidence_digest: Some(first.digest().unwrap()),
            checkpoint_head: first.checkpoint_head,
            state,
            diagnostic_digest: None,
            after_world_digest: None,
            recovery_outcome: None,
            verification: None,
        }
    }

    #[test]
    fn discovers_incomplete_attempt_without_retained_in_memory_key() {
        let path = temp_db("discover");
        let first = armed(context("reserve-a"));
        let second = terminal(&first, AttemptEvidenceState::Applied);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
            journal.append(&first).unwrap();
            journal.append(&second).unwrap();
        }

        let index = SqliteAttemptRecoveryIndex::open_read_only(&path).unwrap();
        let attempts = index.scan_incomplete().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].attempt_key, first.context.attempt_key());
        assert_eq!(attempts[0].last_state, AttemptEvidenceState::Applied);
        assert_eq!(attempts[0].record_count, 2);
        assert_eq!(attempts[0].disposition, LocalEvidenceDisposition::ChargeAsApplied);
        assert!(attempts[0].matches_reservation_v1(&ReservationId("reserve-a".into())));
        cleanup(&path);
    }

    #[test]
    fn proven_not_dispatched_does_not_authorize_post_crash_release() {
        let path = temp_db("no-release");
        let first = armed(context("reserve-b"));
        let second = terminal(&first, AttemptEvidenceState::ProvenNotDispatched);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
            journal.append(&first).unwrap();
            journal.append(&second).unwrap();
        }

        let index = SqliteAttemptRecoveryIndex::open_read_only(&path).unwrap();
        let attempt = index.scan_incomplete().unwrap().pop().unwrap();
        assert_eq!(
            attempt.disposition,
            LocalEvidenceDisposition::NonDispatchClaimNeedsIndependentTrust
        );
        assert_eq!(
            restart_reservation_disposition(ReservationState::Reserved),
            RestartReservationDisposition::NormalizeReservedToOutcomeUnknown
        );
        cleanup(&path);
    }

    #[test]
    fn invalid_semantic_transition_is_rejected_even_when_hash_chain_is_valid() {
        let path = temp_db("state-machine");
        let first = armed(context("reserve-c"));
        let illegal = terminal(&first, AttemptEvidenceState::DispatchArmed);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
            journal.append(&first).unwrap();
            // The underlying v1 record hash accepts a successor DispatchArmed;
            // recovery adds the semantic state-machine validation deliberately.
            journal.append(&illegal).unwrap();
        }

        let index = SqliteAttemptRecoveryIndex::open_read_only(&path).unwrap();
        assert!(matches!(
            index.scan_all().unwrap_err(),
            RecoveryIndexError::InvalidRecordShape
                | RecoveryIndexError::InvalidStateTransition { .. }
        ));
        cleanup(&path);
    }

    #[test]
    fn digest_tampering_is_detected_on_restart_scan() {
        let path = temp_db("tamper");
        let first = armed(context("reserve-d"));
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
            journal.append(&first).unwrap();
        }
        {
            let connection = Connection::open(&path).unwrap();
            connection
                .execute(
                    "UPDATE system_attempt_evidence SET digest = zeroblob(32) WHERE sequence = 0",
                    [],
                )
                .unwrap();
        }

        let index = SqliteAttemptRecoveryIndex::open_read_only(&path).unwrap();
        assert!(matches!(
            index.scan_all().unwrap_err(),
            RecoveryIndexError::RecordDigestMismatch
        ));
        cleanup(&path);
    }

    #[test]
    fn multiple_open_attempts_for_one_reservation_trigger_containment() {
        let path = temp_db("ambiguous");
        let first_a = armed(context("reserve-e"));
        let mut context_b = context("reserve-e");
        context_b.execution_digest = Digest32([77; 32]);
        let first_b = armed(context_b);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
            journal.append(&first_a).unwrap();
            journal.append(&first_b).unwrap();
        }

        let index = SqliteAttemptRecoveryIndex::open_read_only(&path).unwrap();
        assert!(matches!(
            index
                .incomplete_for_reservation(&ReservationId("reserve-e".into()))
                .unwrap_err(),
            RecoveryIndexError::AmbiguousReservationAttempt
        ));
        cleanup(&path);
    }
}
