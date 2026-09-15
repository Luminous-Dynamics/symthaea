// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SQLite exact-CAS persistence for SCI-014 control-plane occurrences.
//!
//! This crate proves storage mechanics only:
//!
//! ```text
//! SQLite mechanics
//!     != deployment-owned store authority
//!     != historical occurrence authority
//!     != current scientific authority
//! ```
//!
//! V1 deliberately performs full checked replay of the append-only occurrence chain on every
//! read snapshot. This is a reference correctness theorem, not a throughput optimization.

#![forbid(unsafe_code)]

use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use symthaea_scientific_view_control_plane_occurrence::wire::ControlPlaneOccurrenceWireError;
use symthaea_scientific_view_control_plane_occurrence::{
    ControlPlaneCommitOperationIdV1, ControlPlaneOccurrenceHeadV1,
    ControlPlaneOccurrenceStoreBindingV1, ControlPlaneOccurrenceStoreV1,
    ControlPlaneStoreCasResultV1, ControlPlaneStoreOperationResolutionV1,
    ProposedControlPlaneOccurrenceV1, RawControlPlaneOccurrenceRecordV1,
};
use symthaea_scientific_view_profile::Commitment32;
use thiserror::Error;

const SQLITE_SCHEMA_REVISION_V1: i64 = 1;
const BUSY_TIMEOUT_MS_V1: i64 = 5_000;

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS control_plane_store_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_revision INTEGER NOT NULL CHECK (schema_revision > 0),
    deployment_id TEXT NOT NULL,
    view_namespace TEXT NOT NULL,
    store_namespace TEXT NOT NULL,
    provisioning_epoch INTEGER NOT NULL CHECK (provisioning_epoch > 0),
    persistence_profile_commitment BLOB NOT NULL CHECK (length(persistence_profile_commitment) = 32),
    store_binding_commitment BLOB NOT NULL CHECK (length(store_binding_commitment) = 32),
    store_instance_binding_commitment BLOB NOT NULL CHECK (length(store_instance_binding_commitment) = 32)
);

CREATE TABLE IF NOT EXISTS control_plane_occurrences (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    occurrence_commitment BLOB NOT NULL UNIQUE CHECK (length(occurrence_commitment) = 32),
    predecessor_occurrence BLOB CHECK (predecessor_occurrence IS NULL OR length(predecessor_occurrence) = 32),
    candidate_transition_commitment BLOB NOT NULL CHECK (length(candidate_transition_commitment) = 32),
    operation_id BLOB NOT NULL UNIQUE CHECK (length(operation_id) = 32),
    occurrence_bytes BLOB NOT NULL,
    store_reference TEXT NOT NULL UNIQUE,
    UNIQUE(sequence, occurrence_commitment),
    FOREIGN KEY(predecessor_occurrence) REFERENCES control_plane_occurrences(occurrence_commitment)
);

CREATE TABLE IF NOT EXISTS control_plane_frontier (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    occurrence_commitment BLOB NOT NULL CHECK (length(occurrence_commitment) = 32),
    FOREIGN KEY(sequence, occurrence_commitment)
        REFERENCES control_plane_occurrences(sequence, occurrence_commitment)
);
"#;

/// Concrete SQLite implementation of the unqualified #3456 occurrence-store contract.
pub struct SqliteControlPlaneOccurrenceStoreV1 {
    connection: Connection,
    binding: ControlPlaneOccurrenceStoreBindingV1,
    store_instance_binding_commitment: Commitment32,
    expected_schema: Vec<SqliteSchemaObjectV1>,
    allow_first_bootstrap: bool,
}

impl SqliteControlPlaneOccurrenceStoreV1 {
    /// Open/create one store under an exact configured occurrence-store identity.
    ///
    /// `store_instance_binding_commitment` is mechanical instance identity only. Passing it here
    /// does not make this SQLite file deployment-authorized; #3465 owns that separate theorem.
    pub fn open(
        path: impl AsRef<Path>,
        binding: ControlPlaneOccurrenceStoreBindingV1,
        store_instance_binding_commitment: Commitment32,
    ) -> Result<Self, SqliteControlPlaneOccurrenceStoreError> {
        if store_instance_binding_commitment.is_zero() {
            return Err(SqliteControlPlaneOccurrenceStoreError::ZeroStoreInstanceBinding);
        }

        let expected_schema = expected_schema_v1()?;
        let path = path.as_ref();
        let allow_first_bootstrap = atomically_claim_new_database_path(path)?;
        let connection = Connection::open(path)?;

        if allow_first_bootstrap {
            configure_fresh_database(&connection)?;
        } else {
            validate_existing_store_before_configuration(
                &connection,
                &expected_schema,
                &binding,
                store_instance_binding_commitment,
            )?;
            configure_existing_database(&connection)?;
        }

        let mut store = Self {
            connection,
            binding,
            store_instance_binding_commitment,
            expected_schema,
            allow_first_bootstrap,
        };
        store.initialize_or_validate_metadata()?;
        Ok(store)
    }

    pub fn configured_binding(&self) -> &ControlPlaneOccurrenceStoreBindingV1 {
        &self.binding
    }

    pub const fn store_instance_binding_commitment(&self) -> Commitment32 {
        self.store_instance_binding_commitment
    }

    fn initialize_or_validate_metadata(
        &mut self,
    ) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;

        if self.allow_first_bootstrap {
            initialize_fresh_schema(&transaction, &self.expected_schema)?;
        } else {
            validate_schema_contract(&transaction, &self.expected_schema)?;
        }

        let existing = read_metadata(&transaction)?;
        match (self.allow_first_bootstrap, existing) {
            (true, None) => {
                let occurrence_count: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM control_plane_occurrences",
                    [],
                    |row| row.get(0),
                )?;
                let frontier_count: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM control_plane_frontier",
                    [],
                    |row| row.get(0),
                )?;
                if occurrence_count != 0 || frontier_count != 0 {
                    return Err(SqliteControlPlaneOccurrenceStoreError::FreshBootstrapContested);
                }

                let sequence_epoch = i64::try_from(self.binding.provisioning_epoch())
                    .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
                let inserted = transaction.execute(
                    "INSERT INTO control_plane_store_metadata(
                        singleton, schema_revision, deployment_id, view_namespace, store_namespace,
                        provisioning_epoch, persistence_profile_commitment,
                        store_binding_commitment, store_instance_binding_commitment
                     ) VALUES(1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        SQLITE_SCHEMA_REVISION_V1,
                        self.binding.deployment_id(),
                        self.binding.view_namespace(),
                        self.binding.store_namespace(),
                        sequence_epoch,
                        blob(self.binding.persistence_profile_commitment()),
                        blob(self.binding.commitment()),
                        blob(self.store_instance_binding_commitment),
                    ],
                )?;
                if inserted != 1 {
                    return Err(SqliteControlPlaneOccurrenceStoreError::AtomicWriteInvariant);
                }
            }
            (true, Some(_)) => {
                return Err(SqliteControlPlaneOccurrenceStoreError::FreshBootstrapContested);
            }
            (false, Some(metadata)) => validate_metadata_values(
                &metadata,
                &self.binding,
                self.store_instance_binding_commitment,
            )?,
            (false, None) => {
                let occurrence_count: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM control_plane_occurrences",
                    [],
                    |row| row.get(0),
                )?;
                let frontier_count: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM control_plane_frontier",
                    [],
                    |row| row.get(0),
                )?;
                if occurrence_count != 0 || frontier_count != 0 {
                    return Err(
                        SqliteControlPlaneOccurrenceStoreError::UnprovisionedStoreContainsState {
                            occurrence_count,
                            frontier_count,
                        },
                    );
                }
                return Err(SqliteControlPlaneOccurrenceStoreError::ExistingStoreUnprovisioned);
            }
        }

        transaction.commit()?;
        Ok(())
    }

    fn validate_binding_argument(
        &self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
    ) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
        if binding != &self.binding {
            return Err(SqliteControlPlaneOccurrenceStoreError::ConfiguredBindingMismatch);
        }
        Ok(())
    }
}

impl ControlPlaneOccurrenceStoreV1 for SqliteControlPlaneOccurrenceStoreV1 {
    type Error = SqliteControlPlaneOccurrenceStoreError;

    fn load_frontier(
        &self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
    ) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> {
        self.validate_binding_argument(binding)?;

        // `unchecked_transaction` accepts &Connection and still gives us one coherent SQLite read
        // snapshot. There are no nested transactions in this store API.
        let transaction = self.connection.unchecked_transaction()?;
        validate_schema_contract(&transaction, &self.expected_schema)?;
        validate_metadata(
            &transaction,
            &self.binding,
            self.store_instance_binding_commitment,
        )?;
        let records = load_validated_chain_and_frontier(&transaction, &self.binding)?;
        let result = records.last().cloned();
        transaction.commit()?;
        Ok(result)
    }

    fn compare_and_swap(
        &mut self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
        expected: Option<ControlPlaneOccurrenceHeadV1>,
        proposed: &ProposedControlPlaneOccurrenceV1,
    ) -> Result<ControlPlaneStoreCasResultV1, Self::Error> {
        self.validate_binding_argument(binding)?;
        if proposed.store_binding_commitment() != self.binding.commitment() {
            return Err(SqliteControlPlaneOccurrenceStoreError::OccurrenceBindingMismatch);
        }

        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        validate_schema_contract(&transaction, &self.expected_schema)?;
        validate_metadata(
            &transaction,
            &self.binding,
            self.store_instance_binding_commitment,
        )?;
        let records = load_validated_chain_and_frontier(&transaction, &self.binding)?;

        if let Some(existing) = records
            .iter()
            .find(|record| record.occurrence().operation_id() == proposed.operation_id())
        {
            if existing.occurrence() != proposed {
                return Err(SqliteControlPlaneOccurrenceStoreError::OperationIdentityConflict);
            }
            let reference = existing.store_reference().to_owned();
            transaction.commit()?;
            return Ok(ControlPlaneStoreCasResultV1::Applied {
                store_reference: reference,
            });
        }

        let actual_frontier = records.last().map(RawControlPlaneOccurrenceRecordV1::head);
        if actual_frontier != expected {
            transaction.rollback()?;
            return Ok(ControlPlaneStoreCasResultV1::Conflict { actual_frontier });
        }

        validate_proposed_successor(actual_frontier, proposed)?;

        let sequence = i64::try_from(proposed.sequence())
            .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
        let predecessor = proposed.predecessor_occurrence().map(blob);
        let store_reference = deterministic_store_reference(&self.binding, proposed);
        let canonical_bytes = proposed.to_canonical_bytes_v1();

        let inserted = transaction.execute(
            "INSERT INTO control_plane_occurrences(
                sequence, occurrence_commitment, predecessor_occurrence,
                candidate_transition_commitment, operation_id, occurrence_bytes, store_reference
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                sequence,
                blob(proposed.commitment()),
                predecessor,
                blob(proposed.candidate_transition_commitment()),
                blob(proposed.operation_id().commitment()),
                canonical_bytes,
                store_reference,
            ],
        )?;
        if inserted != 1 {
            return Err(SqliteControlPlaneOccurrenceStoreError::AtomicWriteInvariant);
        }

        match expected {
            None => {
                let inserted = transaction.execute(
                    "INSERT INTO control_plane_frontier(singleton, sequence, occurrence_commitment)
                     VALUES(1, ?1, ?2)",
                    params![sequence, blob(proposed.commitment())],
                )?;
                if inserted != 1 {
                    return Err(SqliteControlPlaneOccurrenceStoreError::AtomicWriteInvariant);
                }
            }
            Some(previous) => {
                let previous_sequence = i64::try_from(previous.sequence())
                    .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
                let updated = transaction.execute(
                    "UPDATE control_plane_frontier
                     SET sequence = ?1, occurrence_commitment = ?2
                     WHERE singleton = 1 AND sequence = ?3 AND occurrence_commitment = ?4",
                    params![
                        sequence,
                        blob(proposed.commitment()),
                        previous_sequence,
                        blob(previous.occurrence_commitment()),
                    ],
                )?;
                if updated != 1 {
                    return Err(SqliteControlPlaneOccurrenceStoreError::AtomicWriteInvariant);
                }
            }
        }

        transaction.commit()?;

        // Do not trust the transaction's in-memory candidate. Re-open a coherent read snapshot and
        // prove that the deterministic operation is durably represented by the exact occurrence.
        match self.resolve_operation(&self.binding, proposed.operation_id())? {
            ControlPlaneStoreOperationResolutionV1::Found(record)
                if record.occurrence() == proposed && record.store_reference() == store_reference =>
            {
                Ok(ControlPlaneStoreCasResultV1::Applied { store_reference })
            }
            _ => Err(SqliteControlPlaneOccurrenceStoreError::DurableReadbackMismatch),
        }
    }

    fn resolve_operation(
        &self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
        operation_id: ControlPlaneCommitOperationIdV1,
    ) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> {
        self.validate_binding_argument(binding)?;

        // The operation result and returned frontier are derived from one SQLite snapshot. This is
        // required by #3472; separately timed reads cannot establish `ProvenAbsent { frontier }`.
        let transaction = self.connection.unchecked_transaction()?;
        validate_schema_contract(&transaction, &self.expected_schema)?;
        validate_metadata(
            &transaction,
            &self.binding,
            self.store_instance_binding_commitment,
        )?;
        let records = load_validated_chain_and_frontier(&transaction, &self.binding)?;

        let mut matches = records
            .iter()
            .filter(|record| record.occurrence().operation_id() == operation_id);
        let first = matches.next().cloned();
        if matches.next().is_some() {
            return Err(SqliteControlPlaneOccurrenceStoreError::DuplicateOperationIdentity);
        }
        let frontier = records.last().map(RawControlPlaneOccurrenceRecordV1::head);

        let result = match first {
            Some(record) => ControlPlaneStoreOperationResolutionV1::Found(record),
            None => ControlPlaneStoreOperationResolutionV1::ProvenAbsent {
                current_frontier: frontier,
            },
        };
        transaction.commit()?;
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SqliteSchemaObjectV1 {
    object_type: String,
    name: String,
    table_name: String,
    sql: String,
}

#[derive(Debug, Clone)]
struct StoreMetadataRow {
    schema_revision: i64,
    deployment_id: String,
    view_namespace: String,
    store_namespace: String,
    provisioning_epoch: i64,
    persistence_profile_commitment: Vec<u8>,
    store_binding_commitment: Vec<u8>,
    store_instance_binding_commitment: Vec<u8>,
}

#[derive(Debug)]
struct StoredOccurrenceRow {
    sequence: i64,
    occurrence_commitment: Vec<u8>,
    predecessor_occurrence: Option<Vec<u8>>,
    candidate_transition_commitment: Vec<u8>,
    operation_id: Vec<u8>,
    occurrence_bytes: Vec<u8>,
    store_reference: String,
}

fn atomically_claim_new_database_path(
    path: &Path,
) -> Result<bool, SqliteControlPlaneOccurrenceStoreError> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => {
            drop(file);
            Ok(true)
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(SqliteControlPlaneOccurrenceStoreError::Io(error)),
    }
}

fn configure_fresh_database(
    connection: &Connection,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    connection.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=FULL;
         PRAGMA foreign_keys=ON;
         PRAGMA busy_timeout=5000;",
    )?;
    validate_connection_profile(connection)
}

fn configure_existing_database(
    connection: &Connection,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let journal_mode: String = connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        return Err(SqliteControlPlaneOccurrenceStoreError::DurabilityProfileMismatch);
    }

    // These settings are connection-local. Unlike journal_mode=WAL, they do not adopt or rewrite
    // the persistent file-format state of a pre-existing database.
    connection.execute_batch(
        "PRAGMA synchronous=FULL;
         PRAGMA foreign_keys=ON;
         PRAGMA busy_timeout=5000;",
    )?;
    validate_connection_profile(connection)
}

fn validate_connection_profile(
    connection: &Connection,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let journal_mode: String = connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    let synchronous: i64 = connection.query_row("PRAGMA synchronous", [], |row| row.get(0))?;
    let foreign_keys: i64 = connection.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
    let busy_timeout: i64 = connection.query_row("PRAGMA busy_timeout", [], |row| row.get(0))?;

    if !journal_mode.eq_ignore_ascii_case("wal")
        || synchronous != 2
        || foreign_keys != 1
        || busy_timeout != BUSY_TIMEOUT_MS_V1
    {
        return Err(SqliteControlPlaneOccurrenceStoreError::DurabilityProfileMismatch);
    }
    Ok(())
}

fn validate_existing_store_before_configuration(
    connection: &Connection,
    expected_schema: &[SqliteSchemaObjectV1],
    binding: &ControlPlaneOccurrenceStoreBindingV1,
    store_instance_binding_commitment: Commitment32,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let existing_schema = read_application_schema(connection)?;
    if existing_schema.is_empty() {
        return Err(SqliteControlPlaneOccurrenceStoreError::ExistingStoreUnprovisioned);
    }
    if existing_schema != expected_schema {
        return Err(SqliteControlPlaneOccurrenceStoreError::SchemaContractMismatch);
    }

    match read_metadata(connection)? {
        Some(metadata) => {
            validate_metadata_values(&metadata, binding, store_instance_binding_commitment)
        }
        None => {
            let occurrence_count: i64 = connection.query_row(
                "SELECT COUNT(*) FROM control_plane_occurrences",
                [],
                |row| row.get(0),
            )?;
            let frontier_count: i64 = connection.query_row(
                "SELECT COUNT(*) FROM control_plane_frontier",
                [],
                |row| row.get(0),
            )?;
            if occurrence_count != 0 || frontier_count != 0 {
                return Err(
                    SqliteControlPlaneOccurrenceStoreError::UnprovisionedStoreContainsState {
                        occurrence_count,
                        frontier_count,
                    },
                );
            }
            Err(SqliteControlPlaneOccurrenceStoreError::ExistingStoreUnprovisioned)
        }
    }
}

fn expected_schema_v1() -> Result<Vec<SqliteSchemaObjectV1>, SqliteControlPlaneOccurrenceStoreError> {
    let connection = Connection::open_in_memory()?;
    connection.execute_batch(SCHEMA_V1)?;
    read_application_schema(&connection)
}

fn read_application_schema(
    connection: &Connection,
) -> Result<Vec<SqliteSchemaObjectV1>, SqliteControlPlaneOccurrenceStoreError> {
    let mut statement = connection.prepare(
        "SELECT type, name, tbl_name, COALESCE(sql, '')
         FROM sqlite_schema
         WHERE name NOT GLOB 'sqlite_*'
         ORDER BY type ASC, name ASC, tbl_name ASC, sql ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(SqliteSchemaObjectV1 {
            object_type: row.get(0)?,
            name: row.get(1)?,
            table_name: row.get(2)?,
            sql: row.get(3)?,
        })
    })?;

    let mut schema = Vec::new();
    for row in rows {
        schema.push(row?);
    }
    Ok(schema)
}

fn initialize_fresh_schema(
    connection: &Connection,
    expected_schema: &[SqliteSchemaObjectV1],
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let existing_schema = read_application_schema(connection)?;
    if !existing_schema.is_empty() {
        return Err(SqliteControlPlaneOccurrenceStoreError::FreshBootstrapContested);
    }
    connection.execute_batch(SCHEMA_V1)?;
    validate_schema_contract(connection, expected_schema)
}

fn validate_schema_contract(
    connection: &Connection,
    expected_schema: &[SqliteSchemaObjectV1],
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let actual_schema = read_application_schema(connection)?;
    if actual_schema != expected_schema {
        return Err(SqliteControlPlaneOccurrenceStoreError::SchemaContractMismatch);
    }
    Ok(())
}

fn read_metadata(
    connection: &Connection,
) -> Result<Option<StoreMetadataRow>, SqliteControlPlaneOccurrenceStoreError> {
    connection
        .query_row(
            "SELECT schema_revision, deployment_id, view_namespace, store_namespace,
                    provisioning_epoch, persistence_profile_commitment,
                    store_binding_commitment, store_instance_binding_commitment
             FROM control_plane_store_metadata WHERE singleton = 1",
            [],
            |row| {
                Ok(StoreMetadataRow {
                    schema_revision: row.get(0)?,
                    deployment_id: row.get(1)?,
                    view_namespace: row.get(2)?,
                    store_namespace: row.get(3)?,
                    provisioning_epoch: row.get(4)?,
                    persistence_profile_commitment: row.get(5)?,
                    store_binding_commitment: row.get(6)?,
                    store_instance_binding_commitment: row.get(7)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
}

fn validate_metadata(
    connection: &Connection,
    binding: &ControlPlaneOccurrenceStoreBindingV1,
    store_instance_binding_commitment: Commitment32,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let metadata = read_metadata(connection)?
        .ok_or(SqliteControlPlaneOccurrenceStoreError::MissingStoreMetadata)?;
    validate_metadata_values(&metadata, binding, store_instance_binding_commitment)
}

fn validate_metadata_values(
    metadata: &StoreMetadataRow,
    binding: &ControlPlaneOccurrenceStoreBindingV1,
    store_instance_binding_commitment: Commitment32,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    let epoch = i64::try_from(binding.provisioning_epoch())
        .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
    if metadata.schema_revision != SQLITE_SCHEMA_REVISION_V1
        || metadata.deployment_id != binding.deployment_id()
        || metadata.view_namespace != binding.view_namespace()
        || metadata.store_namespace != binding.store_namespace()
        || metadata.provisioning_epoch != epoch
        || decode_commitment(
            &metadata.persistence_profile_commitment,
            "metadata.persistence_profile_commitment",
        )? != binding.persistence_profile_commitment()
        || decode_commitment(
            &metadata.store_binding_commitment,
            "metadata.store_binding_commitment",
        )? != binding.commitment()
        || decode_commitment(
            &metadata.store_instance_binding_commitment,
            "metadata.store_instance_binding_commitment",
        )? != store_instance_binding_commitment
    {
        return Err(SqliteControlPlaneOccurrenceStoreError::StoreMetadataMismatch);
    }
    Ok(())
}

fn load_validated_chain_and_frontier(
    connection: &Connection,
    binding: &ControlPlaneOccurrenceStoreBindingV1,
) -> Result<Vec<RawControlPlaneOccurrenceRecordV1>, SqliteControlPlaneOccurrenceStoreError> {
    let mut statement = connection.prepare(
        "SELECT sequence, occurrence_commitment, predecessor_occurrence,
                candidate_transition_commitment, operation_id, occurrence_bytes, store_reference
         FROM control_plane_occurrences ORDER BY sequence ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(StoredOccurrenceRow {
            sequence: row.get(0)?,
            occurrence_commitment: row.get(1)?,
            predecessor_occurrence: row.get(2)?,
            candidate_transition_commitment: row.get(3)?,
            operation_id: row.get(4)?,
            occurrence_bytes: row.get(5)?,
            store_reference: row.get(6)?,
        })
    })?;

    let mut records = Vec::new();
    let mut previous_commitment = None;
    for (index, row) in rows.enumerate() {
        let row = row?;
        let record = decode_occurrence_row(binding, row)?;
        let expected_sequence = u64::try_from(index + 1)
            .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
        if record.occurrence().sequence() != expected_sequence {
            return Err(SqliteControlPlaneOccurrenceStoreError::NonContiguousOccurrenceSequence);
        }
        if record.occurrence().predecessor_occurrence() != previous_commitment {
            return Err(SqliteControlPlaneOccurrenceStoreError::OccurrencePredecessorMismatch);
        }
        previous_commitment = Some(record.occurrence().commitment());
        records.push(record);
    }
    drop(statement);

    let stored_frontier = connection
        .query_row(
            "SELECT sequence, occurrence_commitment FROM control_plane_frontier WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()?;

    match (records.last(), stored_frontier) {
        (None, None) => {}
        (None, Some(_)) => return Err(SqliteControlPlaneOccurrenceStoreError::FrontierMismatch),
        (Some(_), None) => return Err(SqliteControlPlaneOccurrenceStoreError::MissingFrontier),
        (Some(last), Some((sequence, commitment))) => {
            let sequence = u64::try_from(sequence)
                .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
            let commitment = decode_commitment(&commitment, "frontier.occurrence_commitment")?;
            if sequence != last.occurrence().sequence()
                || commitment != last.occurrence().commitment()
            {
                return Err(SqliteControlPlaneOccurrenceStoreError::FrontierMismatch);
            }
        }
    }

    Ok(records)
}

fn decode_occurrence_row(
    binding: &ControlPlaneOccurrenceStoreBindingV1,
    row: StoredOccurrenceRow,
) -> Result<RawControlPlaneOccurrenceRecordV1, SqliteControlPlaneOccurrenceStoreError> {
    let sequence = u64::try_from(row.sequence)
        .map_err(|_| SqliteControlPlaneOccurrenceStoreError::IntegerOutOfRange)?;
    let stored_occurrence = decode_commitment(&row.occurrence_commitment, "occurrence.commitment")?;
    let stored_predecessor = row
        .predecessor_occurrence
        .as_deref()
        .map(|value| decode_commitment(value, "occurrence.predecessor"))
        .transpose()?;
    let stored_candidate = decode_commitment(
        &row.candidate_transition_commitment,
        "occurrence.candidate_transition_commitment",
    )?;
    let stored_operation = decode_commitment(&row.operation_id, "occurrence.operation_id")?;

    let occurrence = ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(
        binding,
        &row.occurrence_bytes,
    )?;

    if occurrence.sequence() != sequence
        || occurrence.commitment() != stored_occurrence
        || occurrence.predecessor_occurrence() != stored_predecessor
        || occurrence.candidate_transition_commitment() != stored_candidate
        || occurrence.operation_id().commitment() != stored_operation
    {
        return Err(SqliteControlPlaneOccurrenceStoreError::IndexedOccurrenceMismatch);
    }

    let expected_reference = deterministic_store_reference(binding, &occurrence);
    if row.store_reference != expected_reference {
        return Err(SqliteControlPlaneOccurrenceStoreError::StoreReferenceMismatch);
    }

    RawControlPlaneOccurrenceRecordV1::new_unqualified(occurrence, row.store_reference)
        .map_err(Into::into)
}

fn validate_proposed_successor(
    actual_frontier: Option<ControlPlaneOccurrenceHeadV1>,
    proposed: &ProposedControlPlaneOccurrenceV1,
) -> Result<(), SqliteControlPlaneOccurrenceStoreError> {
    match actual_frontier {
        None => {
            if proposed.sequence() != 1 || proposed.predecessor_occurrence().is_some() {
                return Err(SqliteControlPlaneOccurrenceStoreError::InvalidProposedSuccessor);
            }
        }
        Some(previous) => {
            let expected_sequence = previous
                .sequence()
                .checked_add(1)
                .ok_or(SqliteControlPlaneOccurrenceStoreError::SequenceOverflow)?;
            if proposed.sequence() != expected_sequence
                || proposed.predecessor_occurrence() != Some(previous.occurrence_commitment())
            {
                return Err(SqliteControlPlaneOccurrenceStoreError::InvalidProposedSuccessor);
            }
        }
    }
    Ok(())
}

fn deterministic_store_reference(
    binding: &ControlPlaneOccurrenceStoreBindingV1,
    occurrence: &ProposedControlPlaneOccurrenceV1,
) -> String {
    format!(
        "sqlite-control-plane:v1:{}:{}:{}",
        hex(binding.commitment()),
        occurrence.sequence(),
        hex(occurrence.commitment())
    )
}

fn blob(commitment: Commitment32) -> Vec<u8> {
    commitment.as_bytes().to_vec()
}

fn decode_commitment(
    bytes: &[u8],
    field: &'static str,
) -> Result<Commitment32, SqliteControlPlaneOccurrenceStoreError> {
    let value: [u8; 32] = bytes
        .try_into()
        .map_err(|_| SqliteControlPlaneOccurrenceStoreError::MalformedCommitment { field })?;
    Ok(Commitment32::from_bytes(value))
}

fn hex(commitment: Commitment32) -> String {
    let mut output = String::with_capacity(64);
    for byte in commitment.as_bytes() {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Debug, Error)]
pub enum SqliteControlPlaneOccurrenceStoreError {
    #[error("SQLite occurrence-store file operation failed: {0}")]
    Io(#[source] std::io::Error),
    #[error("SQLite occurrence-store operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("control-plane occurrence wire validation failed: {0}")]
    Wire(#[from] ControlPlaneOccurrenceWireError),
    #[error("control-plane occurrence structural validation failed: {0}")]
    Occurrence(
        #[from]
        symthaea_scientific_view_control_plane_occurrence::ControlPlaneOccurrenceError,
    ),
    #[error("store-instance binding commitment must be nonzero")]
    ZeroStoreInstanceBinding,
    #[error("pre-existing SQLite file is not a provisioned R2 occurrence store")]
    ExistingStoreUnprovisioned,
    #[error("fresh SQLite occurrence-store bootstrap was concurrently contested")]
    FreshBootstrapContested,
    #[error("SQLite occurrence-store durability profile does not match V1")]
    DurabilityProfileMismatch,
    #[error("SQLite application schema does not match exact V1 contract")]
    SchemaContractMismatch,
    #[error("configured occurrence-store binding does not match this SQLite store")]
    ConfiguredBindingMismatch,
    #[error("proposed occurrence binding does not match this SQLite store")]
    OccurrenceBindingMismatch,
    #[error("SQLite store metadata is missing")]
    MissingStoreMetadata,
    #[error("SQLite store metadata does not match configured identity")]
    StoreMetadataMismatch,
    #[error("SQLite metadata is absent but durable store state already exists: occurrences={occurrence_count}, frontier_rows={frontier_count}")]
    UnprovisionedStoreContainsState {
        occurrence_count: i64,
        frontier_count: i64,
    },
    #[error("integer cannot be represented by SQLite INTEGER / host usize")]
    IntegerOutOfRange,
    #[error("malformed 32-byte commitment in {field}")]
    MalformedCommitment { field: &'static str },
    #[error("occurrence sequence is not contiguous from genesis")]
    NonContiguousOccurrenceSequence,
    #[error("occurrence predecessor does not equal the immediately preceding occurrence")]
    OccurrencePredecessorMismatch,
    #[error("SQLite indexed occurrence fields disagree with canonical checked bytes")]
    IndexedOccurrenceMismatch,
    #[error("SQLite store reference disagrees with deterministic occurrence identity")]
    StoreReferenceMismatch,
    #[error("SQLite frontier is missing for a nonempty occurrence chain")]
    MissingFrontier,
    #[error("SQLite frontier disagrees with the validated occurrence chain")]
    FrontierMismatch,
    #[error("deterministic operation id maps to a different occurrence")]
    OperationIdentityConflict,
    #[error("multiple occurrences resolved to one deterministic operation id")]
    DuplicateOperationIdentity,
    #[error("proposed occurrence does not exactly follow the durable frontier")]
    InvalidProposedSuccessor,
    #[error("occurrence sequence overflow")]
    SequenceOverflow,
    #[error("SQLite atomic write invariant failed")]
    AtomicWriteInvariant,
    #[error("post-commit durable operation readback does not equal proposed occurrence")]
    DurableReadbackMismatch,
}
