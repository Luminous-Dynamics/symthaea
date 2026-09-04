// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable, attempt-idempotent sequence reservations for qualification witnesses.
//!
//! The security rule is intentionally conservative:
//!
//! ```text
//! reserve sequence durably
//!   -> verify + sign using exactly that sequence
//!   -> persist the exact attestation durably
//!   -> release it to the caller
//! ```
//!
//! A reservation is never released or reused. A crash can burn availability, but
//! it cannot make an ambiguous sequence available to another notarization. A
//! stable attempt id makes retries idempotent: the same attempt receives the same
//! sequence, while a different binding under that attempt id fails closed.
//!
//! SQLite V1 uses `BEGIN IMMEDIATE`, WAL, and `synchronous=FULL` to serialize
//! writers on one database. The reservation chain detects inconsistent local
//! state, but a coherent rollback of the entire database is outside this crate's
//! threat model; release deployments should externally anchor the returned
//! frontier commitment.

#![deny(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ed25519_dalek::SigningKey;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use symthaea_authority::Digest32;
use symthaea_qualification_witness::QualificationWitnessPolicyV1;
use symthaea_qualification_witness_service::{
    verify_archive_then_sign_v1, QualificationVerifierRuntimePolicyV1,
    ReleaseEvidenceBindingsV1, VerifiedThenSignedQualificationV1,
};
use thiserror::Error;

pub const WITNESS_SEQUENCE_SCHEMA_VERSION: u16 = 1;
pub const MAX_SQLITE_INTEGER: u64 = i64::MAX as u64;

const SQLITE_APPLICATION_ID: i64 = 1_398_363_953; // ASCII "SYW1"
const SQLITE_USER_VERSION: i64 = 1;
const RESERVATION_DOMAIN: &[u8] = b"symthaea.qualification-witness.sequence-reservation.v1\0";
const ATTESTATION_STORAGE_DOMAIN: &[u8] = b"symthaea.qualification-witness.persisted-attestation.v1\0";
const ZERO32: [u8; 32] = [0; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessSequenceAttemptBindingV1 {
    pub attempt_id: [u8; 16],
    pub witness_id: [u8; 16],
    pub witness_epoch: u64,
    pub archive_sha256: Digest32,
    pub git_head: [u8; 20],
    pub git_tree: [u8; 20],
    pub verifier_digest: Digest32,
    pub witness_policy_digest: Digest32,
}

impl WitnessSequenceAttemptBindingV1 {
    fn validate(self) -> Result<(), WitnessSequenceError> {
        if self.attempt_id == [0; 16]
            || self.witness_id == [0; 16]
            || self.witness_epoch == 0
            || self.witness_epoch > MAX_SQLITE_INTEGER
            || self.archive_sha256.0 == ZERO32
            || self.git_head == [0; 20]
            || self.git_tree == [0; 20]
            || self.verifier_digest.0 == ZERO32
            || self.witness_policy_digest.0 == ZERO32
        {
            return Err(WitnessSequenceError::InvalidAttemptBinding);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableWitnessAttemptStateV1 {
    Reserved,
    Signed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessSequenceReservationV1 {
    pub sequence: u64,
    pub reservation_digest: Digest32,
    pub state: DurableWitnessAttemptStateV1,
    pub acceptance_digest: Option<Digest32>,
    pub attestation_digest: Option<Digest32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessSequenceFrontierV1 {
    pub high_watermark: u64,
    pub reservation_head: Digest32,
}

#[derive(Debug)]
pub struct DurableVerifiedThenSignedQualificationV1 {
    pub verified: VerifiedThenSignedQualificationV1,
    attempt_id: [u8; 16],
    sequence: u64,
    reservation_digest: Digest32,
    attestation_digest: Digest32,
}

impl DurableVerifiedThenSignedQualificationV1 {
    pub fn attempt_id(&self) -> [u8; 16] {
        self.attempt_id
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn reservation_digest(&self) -> Digest32 {
        self.reservation_digest
    }

    pub fn attestation_digest(&self) -> Digest32 {
        self.attestation_digest
    }
}

/// Request object for one durable notarization attempt. The same `attempt_id`
/// must be reused when retrying the same logical request after a crash.
pub struct DurableWitnessNotarizationRequestV1<'a> {
    pub attempt_id: [u8; 16],
    pub runtime_policy: &'a QualificationVerifierRuntimePolicyV1,
    pub witness_policy: &'a QualificationWitnessPolicyV1,
    pub witness_id: [u8; 16],
    pub signing_key: &'a SigningKey,
    pub archive_path: &'a Path,
    pub release_bindings: ReleaseEvidenceBindingsV1,
}

/// File-backed SQLite sequence store. A new connection is opened for each
/// operation so independent processes/connections contend through SQLite's
/// transactional writer lock rather than an in-process mutex.
#[derive(Debug, Clone)]
pub struct SqliteWitnessSequenceStore {
    path: PathBuf,
}

impl SqliteWitnessSequenceStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WitnessSequenceError> {
        let path = canonical_store_target(path.as_ref())?;
        let store = Self { path };
        let conn = store.connect()?;
        initialize_schema(&conn)?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn connect(&self) -> Result<Connection, WitnessSequenceError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW
            | OpenFlags::SQLITE_OPEN_EXRESCODE;
        let conn = Connection::open_with_flags(&self.path, flags)?;
        conn.busy_timeout(Duration::from_secs(10))?;
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(WitnessSequenceError::DurabilityConfiguration);
        }
        conn.execute_batch(
            "PRAGMA synchronous=FULL;\n\
             PRAGMA foreign_keys=ON;\n\
             PRAGMA trusted_schema=OFF;",
        )?;
        Ok(conn)
    }

    pub fn reserve_attempt(
        &self,
        binding: WitnessSequenceAttemptBindingV1,
    ) -> Result<WitnessSequenceReservationV1, WitnessSequenceError> {
        binding.validate()?;
        let mut conn = self.connect()?;
        initialize_schema(&conn)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        if let Some(existing) = load_attempt(&tx, binding.witness_id, binding.attempt_id)? {
            verify_existing_binding(&existing, binding)?;
            let reservation = existing.as_reservation()?;
            tx.commit()?;
            return Ok(reservation);
        }

        let frontier = load_frontier(&tx, binding.witness_id)?;
        let (high_watermark, previous_head) = match frontier {
            Some(frontier) => (frontier.high_watermark, frontier.reservation_head),
            None => (0, Digest32(ZERO32)),
        };
        if high_watermark >= MAX_SQLITE_INTEGER {
            return Err(WitnessSequenceError::SequenceExhausted);
        }
        let sequence = high_watermark
            .checked_add(1)
            .ok_or(WitnessSequenceError::SequenceExhausted)?;
        let reservation_digest = reservation_digest(binding, sequence, previous_head)?;

        tx.execute(
            "INSERT INTO witness_sequence_attempts (\n\
                witness_id, attempt_id, sequence, witness_epoch, archive_sha256, git_head, git_tree,\n\
                verifier_digest, witness_policy_digest, previous_reservation_digest, reservation_digest,\n\
                state, acceptance_digest, attestation_digest, attestation_json\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 1, NULL, NULL, NULL)",
            params![
                &binding.witness_id[..],
                &binding.attempt_id[..],
                to_sql_integer(sequence)?,
                to_sql_integer(binding.witness_epoch)?,
                &binding.archive_sha256.0[..],
                &binding.git_head[..],
                &binding.git_tree[..],
                &binding.verifier_digest.0[..],
                &binding.witness_policy_digest.0[..],
                &previous_head.0[..],
                &reservation_digest.0[..],
            ],
        )?;

        match frontier {
            None => {
                tx.execute(
                    "INSERT INTO witness_sequence_frontier (witness_id, high_watermark, reservation_head)\n\
                     VALUES (?1, ?2, ?3)",
                    params![
                        &binding.witness_id[..],
                        to_sql_integer(sequence)?,
                        &reservation_digest.0[..]
                    ],
                )?;
            }
            Some(previous) => {
                let changed = tx.execute(
                    "UPDATE witness_sequence_frontier\n\
                     SET high_watermark=?1, reservation_head=?2\n\
                     WHERE witness_id=?3 AND high_watermark=?4 AND reservation_head=?5",
                    params![
                        to_sql_integer(sequence)?,
                        &reservation_digest.0[..],
                        &binding.witness_id[..],
                        to_sql_integer(previous.high_watermark)?,
                        &previous.reservation_head.0[..]
                    ],
                )?;
                if changed != 1 {
                    return Err(WitnessSequenceError::FrontierConflict);
                }
            }
        }

        tx.commit()?;
        Ok(WitnessSequenceReservationV1 {
            sequence,
            reservation_digest,
            state: DurableWitnessAttemptStateV1::Reserved,
            acceptance_digest: None,
            attestation_digest: None,
        })
    }

    pub fn frontier(
        &self,
        witness_id: [u8; 16],
    ) -> Result<Option<WitnessSequenceFrontierV1>, WitnessSequenceError> {
        if witness_id == [0; 16] {
            return Err(WitnessSequenceError::InvalidAttemptBinding);
        }
        let conn = self.connect()?;
        initialize_schema(&conn)?;
        load_frontier(&conn, witness_id)
    }

    /// Recompute the immutable reservation chain for one witness. This detects
    /// local row/frontier inconsistency, sequence gaps, and binding mutation.
    /// It cannot detect rollback of an otherwise self-consistent whole database.
    pub fn audit_witness(
        &self,
        witness_id: [u8; 16],
    ) -> Result<Option<WitnessSequenceFrontierV1>, WitnessSequenceError> {
        if witness_id == [0; 16] {
            return Err(WitnessSequenceError::InvalidAttemptBinding);
        }
        let conn = self.connect()?;
        initialize_schema(&conn)?;
        let frontier = load_frontier(&conn, witness_id)?;
        let Some(frontier) = frontier else {
            return Ok(None);
        };

        let mut statement = conn.prepare(
            "SELECT attempt_id, sequence, witness_epoch, archive_sha256, git_head, git_tree,\n\
                    verifier_digest, witness_policy_digest, previous_reservation_digest, reservation_digest,\n\
                    state, acceptance_digest, attestation_digest, attestation_json\n\
             FROM witness_sequence_attempts WHERE witness_id=?1 ORDER BY sequence ASC",
        )?;
        let mut rows = statement.query(params![&witness_id[..]])?;
        let mut expected_sequence = 1u64;
        let mut previous_head = Digest32(ZERO32);
        let mut final_head = Digest32(ZERO32);
        while let Some(row) = rows.next()? {
            let attempt = db_attempt_from_row_with_witness(row, witness_id)?;
            if attempt.sequence != expected_sequence
                || attempt.previous_reservation_digest != previous_head
            {
                return Err(WitnessSequenceError::AuditFailure);
            }
            let expected_digest = reservation_digest(attempt.binding(), attempt.sequence, previous_head)?;
            if expected_digest != attempt.reservation_digest {
                return Err(WitnessSequenceError::AuditFailure);
            }
            attempt.validate_state_fields()?;
            final_head = attempt.reservation_digest;
            previous_head = final_head;
            expected_sequence = expected_sequence
                .checked_add(1)
                .ok_or(WitnessSequenceError::SequenceExhausted)?;
        }
        let counted = expected_sequence - 1;
        if counted != frontier.high_watermark || final_head != frontier.reservation_head {
            return Err(WitnessSequenceError::AuditFailure);
        }
        Ok(Some(frontier))
    }

    fn persist_verified(
        &self,
        binding: WitnessSequenceAttemptBindingV1,
        sequence: u64,
        verified: &VerifiedThenSignedQualificationV1,
    ) -> Result<Digest32, WitnessSequenceError> {
        binding.validate()?;
        if sequence == 0 || sequence > MAX_SQLITE_INTEGER {
            return Err(WitnessSequenceError::SequenceMismatch);
        }
        if verified.archive_sha256() != binding.archive_sha256
            || verified.git_head() != binding.git_head
            || verified.git_tree() != binding.git_tree
            || verified.verifier_digest() != binding.verifier_digest
            || verified.attestation.witness_id != binding.witness_id
            || verified.attestation.witness_epoch != binding.witness_epoch
            || verified.attestation.witness_sequence != sequence
            || verified.attestation.verifier_digest != binding.verifier_digest
            || verified.attestation.policy_digest != binding.witness_policy_digest
            || verified.attestation.acceptance_digest != verified.acceptance_digest()
            || verified.attestation.signature.len() != 64
        {
            return Err(WitnessSequenceError::VerifiedResultMismatch);
        }

        let encoded = serde_json::to_vec(&verified.attestation)?;
        let attestation_digest = stored_attestation_digest(&encoded);
        let acceptance_digest = verified.acceptance_digest();

        let mut conn = self.connect()?;
        initialize_schema(&conn)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let attempt = load_attempt(&tx, binding.witness_id, binding.attempt_id)?
            .ok_or(WitnessSequenceError::AttemptMissing)?;
        verify_existing_binding(&attempt, binding)?;
        if attempt.sequence != sequence {
            return Err(WitnessSequenceError::SequenceMismatch);
        }

        match attempt.state {
            DurableWitnessAttemptStateV1::Reserved => {
                let changed = tx.execute(
                    "UPDATE witness_sequence_attempts\n\
                     SET state=2, acceptance_digest=?1, attestation_digest=?2, attestation_json=?3\n\
                     WHERE witness_id=?4 AND attempt_id=?5 AND sequence=?6 AND state=1",
                    params![
                        &acceptance_digest.0[..],
                        &attestation_digest.0[..],
                        &encoded,
                        &binding.witness_id[..],
                        &binding.attempt_id[..],
                        to_sql_integer(sequence)?
                    ],
                )?;
                if changed != 1 {
                    return Err(WitnessSequenceError::AttemptStateConflict);
                }
            }
            DurableWitnessAttemptStateV1::Signed => {
                if attempt.acceptance_digest != Some(acceptance_digest)
                    || attempt.attestation_digest != Some(attestation_digest)
                    || attempt.attestation_json.as_deref() != Some(encoded.as_slice())
                {
                    return Err(WitnessSequenceError::AttemptStateConflict);
                }
            }
        }
        tx.commit()?;
        Ok(attestation_digest)
    }
}

/// Strong production-shaped orchestration. The same request attempt id must be
/// stable across retries of one logical notarization.
pub fn verify_reserve_sign_persist_v1(
    store: &SqliteWitnessSequenceStore,
    request: DurableWitnessNotarizationRequestV1<'_>,
) -> Result<DurableVerifiedThenSignedQualificationV1, WitnessSequenceError> {
    let DurableWitnessNotarizationRequestV1 {
        attempt_id,
        runtime_policy,
        witness_policy,
        witness_id,
        signing_key,
        archive_path,
        release_bindings,
    } = request;

    if attempt_id == [0; 16]
        || witness_id == [0; 16]
        || release_bindings.archive_sha256.0 == ZERO32
        || release_bindings.git_head == [0; 20]
        || release_bindings.git_tree == [0; 20]
    {
        return Err(WitnessSequenceError::InvalidAttemptBinding);
    }
    let metadata = fs::symlink_metadata(archive_path)?;
    if !metadata.file_type().is_file() {
        return Err(WitnessSequenceError::ArchiveNotRegularFile);
    }

    let verifier_digest = runtime_policy.implementation_digest()?;
    let witness_policy_digest = witness_policy.digest()?;
    if !witness_policy.allowed_verifier_digests.contains(&verifier_digest) {
        return Err(WitnessSequenceError::VerifierNotAllowed);
    }
    let enrolled = witness_policy
        .witnesses
        .iter()
        .find(|witness| witness.witness_id == witness_id)
        .ok_or(WitnessSequenceError::WitnessNotEnrolled)?;
    if enrolled.public_key != signing_key.verifying_key().to_bytes() {
        return Err(WitnessSequenceError::WitnessKeyMismatch);
    }

    let binding = WitnessSequenceAttemptBindingV1 {
        attempt_id,
        witness_id,
        witness_epoch: witness_policy.witness_epoch,
        archive_sha256: release_bindings.archive_sha256,
        git_head: release_bindings.git_head,
        git_tree: release_bindings.git_tree,
        verifier_digest,
        witness_policy_digest,
    };
    let reservation = store.reserve_attempt(binding)?;

    // A verifier rejection after this point intentionally leaves the reservation
    // durable. Retrying the same attempt id reuses the same sequence; another
    // attempt can never claim it.
    let verified = verify_archive_then_sign_v1(
        runtime_policy,
        witness_policy,
        witness_id,
        reservation.sequence,
        signing_key,
        archive_path,
        release_bindings,
    )?;
    let attestation_digest = store.persist_verified(binding, reservation.sequence, &verified)?;

    Ok(DurableVerifiedThenSignedQualificationV1 {
        verified,
        attempt_id,
        sequence: reservation.sequence,
        reservation_digest: reservation.reservation_digest,
        attestation_digest,
    })
}

fn canonical_store_target(path: &Path) -> Result<PathBuf, WitnessSequenceError> {
    if path.as_os_str().is_empty() || path == Path::new(":memory:") {
        return Err(WitnessSequenceError::InvalidStorePath);
    }
    let file_name = path.file_name().ok_or(WitnessSequenceError::InvalidStorePath)?;
    let parent = path
        .parent()
        .filter(|candidate| !candidate.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical_parent = fs::canonicalize(parent)?;
    let resolved = canonical_parent.join(file_name);
    match fs::symlink_metadata(&resolved) {
        Ok(metadata) if metadata.file_type().is_file() => {}
        Ok(_) => return Err(WitnessSequenceError::InvalidStorePath),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(WitnessSequenceError::Io(error)),
    }
    Ok(resolved)
}

fn initialize_schema(conn: &Connection) -> Result<(), WitnessSequenceError> {
    let application_id: i64 = conn.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    let user_version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let names = user_table_names(conn)?;

    if application_id == 0 && user_version == 0 {
        if !names.is_empty() {
            return Err(WitnessSequenceError::SchemaMismatch);
        }
        conn.execute_batch(
            "BEGIN IMMEDIATE;\n\
             PRAGMA application_id=1398363953;\n\
             PRAGMA user_version=1;\n\
             CREATE TABLE witness_sequence_frontier (\n\
                witness_id BLOB PRIMARY KEY NOT NULL CHECK(length(witness_id)=16),\n\
                high_watermark INTEGER NOT NULL CHECK(high_watermark >= 1),\n\
                reservation_head BLOB NOT NULL CHECK(length(reservation_head)=32)\n\
             ) WITHOUT ROWID;\n\
             CREATE TABLE witness_sequence_attempts (\n\
                witness_id BLOB NOT NULL CHECK(length(witness_id)=16),\n\
                attempt_id BLOB NOT NULL CHECK(length(attempt_id)=16),\n\
                sequence INTEGER NOT NULL CHECK(sequence >= 1),\n\
                witness_epoch INTEGER NOT NULL CHECK(witness_epoch >= 1),\n\
                archive_sha256 BLOB NOT NULL CHECK(length(archive_sha256)=32),\n\
                git_head BLOB NOT NULL CHECK(length(git_head)=20),\n\
                git_tree BLOB NOT NULL CHECK(length(git_tree)=20),\n\
                verifier_digest BLOB NOT NULL CHECK(length(verifier_digest)=32),\n\
                witness_policy_digest BLOB NOT NULL CHECK(length(witness_policy_digest)=32),\n\
                previous_reservation_digest BLOB NOT NULL CHECK(length(previous_reservation_digest)=32),\n\
                reservation_digest BLOB NOT NULL CHECK(length(reservation_digest)=32),\n\
                state INTEGER NOT NULL CHECK(state IN (1,2)),\n\
                acceptance_digest BLOB NULL CHECK(acceptance_digest IS NULL OR length(acceptance_digest)=32),\n\
                attestation_digest BLOB NULL CHECK(attestation_digest IS NULL OR length(attestation_digest)=32),\n\
                attestation_json BLOB NULL,\n\
                PRIMARY KEY(witness_id, attempt_id),\n\
                UNIQUE(witness_id, sequence)\n\
             ) WITHOUT ROWID;\n\
             COMMIT;",
        )?;
    } else if application_id != SQLITE_APPLICATION_ID || user_version != SQLITE_USER_VERSION {
        return Err(WitnessSequenceError::SchemaMismatch);
    }

    verify_required_tables(conn)
}

fn user_table_names(conn: &Connection) -> Result<Vec<String>, WitnessSequenceError> {
    let mut statement = conn.prepare(
        "SELECT name FROM sqlite_schema\n\
         WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name ASC",
    )?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(names)
}

fn verify_required_tables(conn: &Connection) -> Result<(), WitnessSequenceError> {
    let names = user_table_names(conn)?;
    if names.len() != 2
        || names[0] != "witness_sequence_attempts"
        || names[1] != "witness_sequence_frontier"
    {
        return Err(WitnessSequenceError::SchemaMismatch);
    }
    Ok(())
}

#[derive(Debug)]
struct DbAttempt {
    witness_id: [u8; 16],
    attempt_id: [u8; 16],
    sequence: u64,
    witness_epoch: u64,
    archive_sha256: Digest32,
    git_head: [u8; 20],
    git_tree: [u8; 20],
    verifier_digest: Digest32,
    witness_policy_digest: Digest32,
    previous_reservation_digest: Digest32,
    reservation_digest: Digest32,
    state: DurableWitnessAttemptStateV1,
    acceptance_digest: Option<Digest32>,
    attestation_digest: Option<Digest32>,
    attestation_json: Option<Vec<u8>>,
}

impl DbAttempt {
    fn binding(&self) -> WitnessSequenceAttemptBindingV1 {
        WitnessSequenceAttemptBindingV1 {
            attempt_id: self.attempt_id,
            witness_id: self.witness_id,
            witness_epoch: self.witness_epoch,
            archive_sha256: self.archive_sha256,
            git_head: self.git_head,
            git_tree: self.git_tree,
            verifier_digest: self.verifier_digest,
            witness_policy_digest: self.witness_policy_digest,
        }
    }

    fn validate_state_fields(&self) -> Result<(), WitnessSequenceError> {
        match self.state {
            DurableWitnessAttemptStateV1::Reserved => {
                if self.acceptance_digest.is_some()
                    || self.attestation_digest.is_some()
                    || self.attestation_json.is_some()
                {
                    return Err(WitnessSequenceError::AuditFailure);
                }
            }
            DurableWitnessAttemptStateV1::Signed => {
                let bytes = self
                    .attestation_json
                    .as_ref()
                    .filter(|bytes| !bytes.is_empty())
                    .ok_or(WitnessSequenceError::AuditFailure)?;
                if self.acceptance_digest.is_none()
                    || self.attestation_digest != Some(stored_attestation_digest(bytes))
                {
                    return Err(WitnessSequenceError::AuditFailure);
                }
            }
        }
        Ok(())
    }

    fn as_reservation(&self) -> Result<WitnessSequenceReservationV1, WitnessSequenceError> {
        self.validate_state_fields()?;
        Ok(WitnessSequenceReservationV1 {
            sequence: self.sequence,
            reservation_digest: self.reservation_digest,
            state: self.state,
            acceptance_digest: self.acceptance_digest,
            attestation_digest: self.attestation_digest,
        })
    }
}

struct RawAttempt {
    sequence: i64,
    witness_epoch: i64,
    archive_sha256: Vec<u8>,
    git_head: Vec<u8>,
    git_tree: Vec<u8>,
    verifier_digest: Vec<u8>,
    witness_policy_digest: Vec<u8>,
    previous_reservation_digest: Vec<u8>,
    reservation_digest: Vec<u8>,
    state: i64,
    acceptance_digest: Option<Vec<u8>>,
    attestation_digest: Option<Vec<u8>>,
    attestation_json: Option<Vec<u8>>,
}

fn load_attempt(
    conn: &Connection,
    witness_id: [u8; 16],
    attempt_id: [u8; 16],
) -> Result<Option<DbAttempt>, WitnessSequenceError> {
    conn.query_row(
        "SELECT sequence, witness_epoch, archive_sha256, git_head, git_tree, verifier_digest,\n\
                witness_policy_digest, previous_reservation_digest, reservation_digest, state,\n\
                acceptance_digest, attestation_digest, attestation_json\n\
         FROM witness_sequence_attempts WHERE witness_id=?1 AND attempt_id=?2",
        params![&witness_id[..], &attempt_id[..]],
        |row| {
            Ok(RawAttempt {
                sequence: row.get(0)?,
                witness_epoch: row.get(1)?,
                archive_sha256: row.get(2)?,
                git_head: row.get(3)?,
                git_tree: row.get(4)?,
                verifier_digest: row.get(5)?,
                witness_policy_digest: row.get(6)?,
                previous_reservation_digest: row.get(7)?,
                reservation_digest: row.get(8)?,
                state: row.get(9)?,
                acceptance_digest: row.get(10)?,
                attestation_digest: row.get(11)?,
                attestation_json: row.get(12)?,
            })
        },
    )
    .optional()?
    .map(|raw| db_attempt_from_raw(witness_id, attempt_id, raw))
    .transpose()
}

fn db_attempt_from_raw(
    witness_id: [u8; 16],
    attempt_id: [u8; 16],
    raw: RawAttempt,
) -> Result<DbAttempt, WitnessSequenceError> {
    Ok(DbAttempt {
        witness_id,
        attempt_id,
        sequence: from_sql_integer(raw.sequence)?,
        witness_epoch: from_sql_integer(raw.witness_epoch)?,
        archive_sha256: Digest32(exact_array::<32>(&raw.archive_sha256)?),
        git_head: exact_array::<20>(&raw.git_head)?,
        git_tree: exact_array::<20>(&raw.git_tree)?,
        verifier_digest: Digest32(exact_array::<32>(&raw.verifier_digest)?),
        witness_policy_digest: Digest32(exact_array::<32>(&raw.witness_policy_digest)?),
        previous_reservation_digest: Digest32(exact_array::<32>(&raw.previous_reservation_digest)?),
        reservation_digest: Digest32(exact_array::<32>(&raw.reservation_digest)?),
        state: match raw.state {
            1 => DurableWitnessAttemptStateV1::Reserved,
            2 => DurableWitnessAttemptStateV1::Signed,
            _ => return Err(WitnessSequenceError::AuditFailure),
        },
        acceptance_digest: raw
            .acceptance_digest
            .map(|bytes| exact_array::<32>(&bytes).map(Digest32))
            .transpose()?,
        attestation_digest: raw
            .attestation_digest
            .map(|bytes| exact_array::<32>(&bytes).map(Digest32))
            .transpose()?,
        attestation_json: raw.attestation_json,
    })
}

fn db_attempt_from_row_with_witness(
    row: &rusqlite::Row<'_>,
    witness_id: [u8; 16],
) -> Result<DbAttempt, WitnessSequenceError> {
    let attempt_id: Vec<u8> = row.get(0)?;
    let raw = RawAttempt {
        sequence: row.get(1)?,
        witness_epoch: row.get(2)?,
        archive_sha256: row.get(3)?,
        git_head: row.get(4)?,
        git_tree: row.get(5)?,
        verifier_digest: row.get(6)?,
        witness_policy_digest: row.get(7)?,
        previous_reservation_digest: row.get(8)?,
        reservation_digest: row.get(9)?,
        state: row.get(10)?,
        acceptance_digest: row.get(11)?,
        attestation_digest: row.get(12)?,
        attestation_json: row.get(13)?,
    };
    db_attempt_from_raw(witness_id, exact_array::<16>(&attempt_id)?, raw)
}

fn verify_existing_binding(
    attempt: &DbAttempt,
    binding: WitnessSequenceAttemptBindingV1,
) -> Result<(), WitnessSequenceError> {
    if attempt.binding() != binding {
        return Err(WitnessSequenceError::AttemptBindingConflict);
    }
    let expected = reservation_digest(binding, attempt.sequence, attempt.previous_reservation_digest)?;
    if expected != attempt.reservation_digest {
        return Err(WitnessSequenceError::AuditFailure);
    }
    attempt.validate_state_fields()?;
    Ok(())
}

fn load_frontier(
    conn: &Connection,
    witness_id: [u8; 16],
) -> Result<Option<WitnessSequenceFrontierV1>, WitnessSequenceError> {
    conn.query_row(
        "SELECT high_watermark, reservation_head FROM witness_sequence_frontier WHERE witness_id=?1",
        params![&witness_id[..]],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
    )
    .optional()?
    .map(|(high, head)| {
        Ok(WitnessSequenceFrontierV1 {
            high_watermark: from_sql_integer(high)?,
            reservation_head: Digest32(exact_array::<32>(&head)?),
        })
    })
    .transpose()
}

fn reservation_digest(
    binding: WitnessSequenceAttemptBindingV1,
    sequence: u64,
    previous_head: Digest32,
) -> Result<Digest32, WitnessSequenceError> {
    binding.validate()?;
    if sequence == 0 || sequence > MAX_SQLITE_INTEGER {
        return Err(WitnessSequenceError::SequenceMismatch);
    }
    let mut transcript = Transcript::new(RESERVATION_DOMAIN);
    transcript.u16(WITNESS_SEQUENCE_SCHEMA_VERSION);
    transcript.fixed(&binding.attempt_id);
    transcript.fixed(&binding.witness_id);
    transcript.u64(binding.witness_epoch);
    transcript.u64(sequence);
    transcript.fixed(&binding.archive_sha256.0);
    transcript.fixed(&binding.git_head);
    transcript.fixed(&binding.git_tree);
    transcript.fixed(&binding.verifier_digest.0);
    transcript.fixed(&binding.witness_policy_digest.0);
    transcript.fixed(&previous_head.0);
    Ok(Digest32(transcript.finish()))
}

fn stored_attestation_digest(bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ATTESTATION_STORAGE_DOMAIN);
    hasher.update(bytes);
    Digest32(*hasher.finalize().as_bytes())
}

fn exact_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], WitnessSequenceError> {
    bytes.try_into().map_err(|_| WitnessSequenceError::AuditFailure)
}

fn to_sql_integer(value: u64) -> Result<i64, WitnessSequenceError> {
    i64::try_from(value).map_err(|_| WitnessSequenceError::SequenceExhausted)
}

fn from_sql_integer(value: i64) -> Result<u64, WitnessSequenceError> {
    u64::try_from(value).map_err(|_| WitnessSequenceError::AuditFailure)
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 256);
        bytes.extend_from_slice(domain);
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[derive(Debug, Error)]
pub enum WitnessSequenceError {
    #[error("invalid witness sequence store path")]
    InvalidStorePath,
    #[error("SQLite durability configuration did not enter WAL mode")]
    DurabilityConfiguration,
    #[error("witness sequence database schema/application identity mismatch")]
    SchemaMismatch,
    #[error("invalid witness attempt binding")]
    InvalidAttemptBinding,
    #[error("witness sequence space is exhausted")]
    SequenceExhausted,
    #[error("same attempt id was reused with different security bindings")]
    AttemptBindingConflict,
    #[error("witness reservation frontier changed unexpectedly")]
    FrontierConflict,
    #[error("reserved witness attempt is missing")]
    AttemptMissing,
    #[error("witness sequence does not match durable reservation")]
    SequenceMismatch,
    #[error("durable witness attempt state conflicts with the verified result")]
    AttemptStateConflict,
    #[error("verify-then-sign result disagrees with its durable reservation")]
    VerifiedResultMismatch,
    #[error("local witness reservation chain/frontier audit failed")]
    AuditFailure,
    #[error("evidence verifier implementation is not admitted by witness policy")]
    VerifierNotAllowed,
    #[error("witness id is not enrolled by witness policy")]
    WitnessNotEnrolled,
    #[error("signing key does not match enrolled witness key")]
    WitnessKeyMismatch,
    #[error("qualification archive is not a regular file")]
    ArchiveNotRegularFile,
    #[error("witness service rejected operation: {0}")]
    Service(#[from] symthaea_qualification_witness_service::QualificationWitnessServiceError),
    #[error("witness protocol rejected operation: {0}")]
    Witness(#[from] symthaea_qualification_witness::QualificationWitnessError),
    #[error("SQLite failure: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON persistence failure: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    static NEXT_DB: AtomicU64 = AtomicU64::new(1);

    struct TestDb {
        path: PathBuf,
    }

    impl TestDb {
        fn new() -> Self {
            let id = NEXT_DB.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "symthaea-witness-sequence-{}-{id}.sqlite3",
                std::process::id()
            ));
            let _ = fs::remove_file(&path);
            Self { path }
        }
    }

    impl Drop for TestDb {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_file(format!("{}-wal", self.path.display()));
            let _ = fs::remove_file(format!("{}-shm", self.path.display()));
        }
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

    #[test]
    fn retry_of_same_attempt_reuses_reserved_sequence() {
        let db = TestDb::new();
        let store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let first = store.reserve_attempt(binding(1)).unwrap();
        let second = store.reserve_attempt(binding(1)).unwrap();
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 1);
        assert_eq!(first.reservation_digest, second.reservation_digest);
        assert_eq!(second.state, DurableWitnessAttemptStateV1::Reserved);
        assert_eq!(store.frontier([0x51; 16]).unwrap().unwrap().high_watermark, 1);
    }

    #[test]
    fn same_attempt_id_with_different_binding_fails_closed() {
        let db = TestDb::new();
        let store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        store.reserve_attempt(binding(1)).unwrap();
        let mut changed = binding(1);
        changed.git_tree = [0x99; 20];
        assert!(matches!(
            store.reserve_attempt(changed),
            Err(WitnessSequenceError::AttemptBindingConflict)
        ));
        assert_eq!(store.frontier([0x51; 16]).unwrap().unwrap().high_watermark, 1);
    }

    #[test]
    fn different_attempts_never_reuse_sequence() {
        let db = TestDb::new();
        let store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let one = store.reserve_attempt(binding(1)).unwrap();
        let two = store.reserve_attempt(binding(2)).unwrap();
        assert_eq!((one.sequence, two.sequence), (1, 2));
        let audit = store.audit_witness([0x51; 16]).unwrap().unwrap();
        assert_eq!(audit.high_watermark, 2);
        assert_eq!(audit.reservation_head, two.reservation_digest);
    }

    #[test]
    fn two_connections_serialize_different_attempts() {
        let db = TestDb::new();
        let a = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let b = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let ba = Arc::clone(&barrier);
        let bb = Arc::clone(&barrier);
        let ta = thread::spawn(move || {
            ba.wait();
            a.reserve_attempt(binding(1)).unwrap().sequence
        });
        let tb = thread::spawn(move || {
            bb.wait();
            b.reserve_attempt(binding(2)).unwrap().sequence
        });
        barrier.wait();
        let mut sequences = vec![ta.join().unwrap(), tb.join().unwrap()];
        sequences.sort_unstable();
        assert_eq!(sequences, vec![1, 2]);

        let audit_store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        assert_eq!(
            audit_store.audit_witness([0x51; 16]).unwrap().unwrap().high_watermark,
            2
        );
    }

    #[test]
    fn crash_after_reservation_does_not_release_sequence() {
        let db = TestDb::new();
        let first_process = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let reserved = first_process.reserve_attempt(binding(1)).unwrap();
        drop(first_process);

        let recovered = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        let same = recovered.reserve_attempt(binding(1)).unwrap();
        let next = recovered.reserve_attempt(binding(2)).unwrap();
        assert_eq!(reserved.sequence, 1);
        assert_eq!(same.sequence, 1);
        assert_eq!(next.sequence, 2);
    }

    #[cfg(unix)]
    #[test]
    fn database_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let target = TestDb::new();
        let target_store = SqliteWitnessSequenceStore::open(&target.path).unwrap();
        drop(target_store);
        let link_path = target.path.with_extension("link.sqlite3");
        let _ = fs::remove_file(&link_path);
        symlink(&target.path, &link_path).unwrap();
        assert!(matches!(
            SqliteWitnessSequenceStore::open(&link_path),
            Err(WitnessSequenceError::InvalidStorePath)
        ));
        let _ = fs::remove_file(link_path);
    }

    #[test]
    fn claimed_database_missing_required_table_fails_closed() {
        let db = TestDb::new();
        let store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        drop(store);
        let conn = Connection::open(&db.path).unwrap();
        conn.execute_batch("DROP TABLE witness_sequence_attempts;").unwrap();
        drop(conn);
        assert!(matches!(
            SqliteWitnessSequenceStore::open(&db.path),
            Err(WitnessSequenceError::SchemaMismatch)
        ));
    }

    #[test]
    fn unclaimed_nonempty_database_is_not_adopted() {
        let db = TestDb::new();
        let conn = Connection::open(&db.path).unwrap();
        conn.execute_batch("CREATE TABLE unrelated(value INTEGER);").unwrap();
        drop(conn);
        assert!(matches!(
            SqliteWitnessSequenceStore::open(&db.path),
            Err(WitnessSequenceError::SchemaMismatch)
        ));
    }

    #[test]
    fn local_chain_audit_detects_row_tampering() {
        let db = TestDb::new();
        let store = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        store.reserve_attempt(binding(1)).unwrap();
        store.reserve_attempt(binding(2)).unwrap();
        drop(store);

        let conn = Connection::open(&db.path).unwrap();
        conn.execute(
            "UPDATE witness_sequence_attempts SET archive_sha256=?1 WHERE witness_id=?2 AND sequence=1",
            params![&[0xaa; 32][..], &[0x51; 16][..]],
        )
        .unwrap();
        drop(conn);

        let reopened = SqliteWitnessSequenceStore::open(&db.path).unwrap();
        assert!(matches!(
            reopened.audit_witness([0x51; 16]),
            Err(WitnessSequenceError::AuditFailure)
        ));
    }
}
