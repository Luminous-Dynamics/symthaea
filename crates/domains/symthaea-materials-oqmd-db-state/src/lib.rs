// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical live-MySQL database-state capture after historical OQMD import.
//!
//! A successful mysql client exit is not evidence that a large SQL dump was fully
//! imported. This layer inventories the entire live database and only then allows
//! MAG-DATA-008 to mint a successful import receipt.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_materials_historical_extraction::OqmdExtractionProtocol;
use symthaea_materials_historical_import::{
    HistoricalImportExecutionReceipt, HistoricalImportProfile, bind_successful_import,
};
use symthaea_materials_oqmd_import_executor::{
    MysqlServerPreflightEvidence, OqmdLocalImportPlan, OqmdMysqlImportEvidence,
    execute_server_preflight,
};
use symthaea_materials_schema_inventory::{
    ColumnInventory, ForeignKeyInventory, IndexColumnInventory, IndexInventory,
    MySqlSchemaInventory, SchemaImportBinding, TableInventory, bind_schema_inventory_to_import,
};
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use symthaea_process_capture::{EnvironmentPolicy, ProcessCapture, ProcessSpec, capture_process};
use thiserror::Error;

/// Digest summary for one exact live-database probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatabaseProbeDigest {
    /// Stable probe label.
    pub label: String,
    /// Exact command manifest.
    pub command_manifest_sha256: String,
    /// Exact process capture.
    pub process_capture_sha256: String,
    /// Exact stdout bytes.
    pub stdout_sha256: String,
}

/// Canonical post-import database-state manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatabaseStateManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Exact local import plan.
    pub plan_sha256: String,
    /// Exact mysql import-process evidence.
    pub import_process_evidence_sha256: String,
    /// Exact post-import server preflight.
    pub post_import_preflight_sha256: String,
    /// Canonical schema inventory.
    pub schema_inventory_sha256: String,
    /// Canonical row-count projection.
    pub row_count_inventory_sha256: String,
    /// All live DB probes in deterministic execution order.
    pub probes: Vec<DatabaseProbeDigest>,
}

impl DatabaseStateManifest {
    /// Deterministic manifest identity.
    pub fn manifest_sha256(&self) -> Result<String, DbStateError> {
        if self.schema_version != 1 {
            return Err(DbStateError::UnsupportedManifestSchema(self.schema_version));
        }
        for digest in [
            &self.plan_sha256,
            &self.import_process_evidence_sha256,
            &self.post_import_preflight_sha256,
            &self.schema_inventory_sha256,
            &self.row_count_inventory_sha256,
        ] {
            valid_sha(digest)?;
        }
        if self.probes.is_empty() {
            return Err(DbStateError::EmptyProbeSet);
        }
        let mut labels = BTreeSet::new();
        for probe in &self.probes {
            if probe.label.trim().is_empty() || !labels.insert(probe.label.as_str()) {
                return Err(DbStateError::InvalidProbeLabel(probe.label.clone()));
            }
            valid_sha(&probe.command_manifest_sha256)?;
            valid_sha(&probe.process_capture_sha256)?;
            valid_sha(&probe.stdout_sha256)?;
        }
        Ok(hash_json(self)?)
    }
}

/// Whole-database state plus the import receipt it authorizes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedImportedDatabaseState {
    /// Post-import server identity/configuration evidence.
    pub post_import_preflight: MysqlServerPreflightEvidence,
    /// Canonical whole-database inventory.
    pub inventory: MySqlSchemaInventory,
    /// Canonical state manifest.
    pub state_manifest: DatabaseStateManifest,
    /// Successful MAG-DATA-008 receipt.
    pub import_receipt: HistoricalImportExecutionReceipt,
    /// MAG-DATA-009 inventory binding.
    pub schema_import_binding: SchemaImportBinding,
}

impl VerifiedImportedDatabaseState {
    /// Deterministic identity over all state evidence.
    pub fn evidence_sha256(&self) -> Result<String, DbStateError> {
        self.inventory.validate().map_err(inv_err)?;
        self.state_manifest.manifest_sha256()?;
        self.schema_import_binding.binding_sha256().map_err(inv_err)?;
        Ok(hash_json(self)?)
    }
}

#[derive(Default)]
struct TableBuild {
    engine: String,
    columns: Vec<ColumnInventory>,
    indexes: BTreeMap<String, IndexBuild>,
    fks: BTreeMap<String, ForeignKeyBuild>,
}

struct IndexBuild {
    unique: bool,
    index_type: String,
    columns: Vec<IndexColumnInventory>,
}

struct ForeignKeyBuild {
    referenced_table: String,
    columns: Vec<(u32, String, String)>,
}

/// Capture the entire imported database and mint the first successful import receipt.
pub fn capture_imported_database_state(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    plan: &OqmdLocalImportPlan,
    import: &OqmdMysqlImportEvidence,
) -> Result<VerifiedImportedDatabaseState, DbStateError> {
    plan.validate_contract(protocol, profile, acquisition).map_err(exec_err)?;
    plan.validate_local_artifacts().map_err(exec_err)?;
    let plan_sha = plan.plan_sha256(protocol, profile, acquisition).map_err(exec_err)?;
    if !import.plan_sha256.eq_ignore_ascii_case(&plan_sha) {
        return Err(DbStateError::ImportPlanMismatch);
    }
    if !import.process_success() {
        return Err(DbStateError::ImportProcessDidNotSucceed);
    }
    if import.process.process_capture.stdout_truncated || import.process.process_capture.stderr_truncated {
        return Err(DbStateError::ImportDiagnosticsTruncated);
    }

    // The server must still be the same configured binary after a long import.
    let post_import_preflight = execute_server_preflight(protocol, profile, acquisition, plan)
        .map_err(exec_err)?;
    let post_preflight_sha = post_import_preflight.evidence_sha256().map_err(exec_err)?;

    let mut probes = Vec::new();
    let table_capture = query(plan, &tables_sql(&plan.database_name)?)?;
    probes.push(probe("00-tables", &table_capture)?);
    let table_rows = rows(&table_capture.stdout, 2, false)?;
    if table_rows.is_empty() {
        return Err(DbStateError::EmptyDatabase);
    }

    let mut tables = BTreeMap::<String, TableBuild>::new();
    for row in table_rows {
        let name = hex_text(&row[0])?;
        let engine = hex_text(&row[1])?;
        if name.is_empty() || engine.is_empty() || tables.contains_key(&name) {
            return Err(DbStateError::InvalidTableMetadata(name));
        }
        tables.insert(name, TableBuild { engine, ..TableBuild::default() });
    }

    let column_capture = query(plan, &columns_sql(&plan.database_name)?)?;
    probes.push(probe("01-columns", &column_capture)?);
    for row in rows(&column_capture.stdout, 10, true)? {
        let table = hex_text(&row[0])?;
        let target = tables.get_mut(&table).ok_or_else(|| DbStateError::UnknownTable(table.clone()))?;
        let name = hex_text(&row[2])?;
        let column_type = hex_text(&row[3])?;
        if name.is_empty() || column_type.is_empty() {
            return Err(DbStateError::UnexpectedQueryShape);
        }
        let default_null = bool01(&row[5])?;
        let collation_null = bool01(&row[7])?;
        target.columns.push(ColumnInventory {
            ordinal_position: u32_value(&row[1])?,
            name,
            column_type,
            nullable: bool01(&row[4])?,
            default_repr: if default_null { None } else { Some(hex_text(&row[6])?) },
            collation: if collation_null { None } else { Some(hex_text(&row[8])?) },
            extra: hex_text(&row[9])?,
        });
    }

    let index_capture = query(plan, &indexes_sql(&plan.database_name)?)?;
    probes.push(probe("02-indexes", &index_capture)?);
    for row in rows(&index_capture.stdout, 9, true)? {
        let table = hex_text(&row[0])?;
        let name = hex_text(&row[1])?;
        let index_type = hex_text(&row[3])?;
        let column_name = hex_text(&row[5])?;
        if name.is_empty() || index_type.is_empty() || column_name.is_empty() {
            return Err(DbStateError::UnsupportedIndexShape { table, index: name });
        }
        let collation = hex_text(&row[8])?;
        let descending = match collation.as_str() {
            "" | "A" => false,
            "D" => true,
            _ => return Err(DbStateError::UnexpectedIndexCollation(collation)),
        };
        let unique = !bool01(&row[2])?;
        let target = tables.get_mut(&table).ok_or_else(|| DbStateError::UnknownTable(table.clone()))?;
        let index = target.indexes.entry(name.clone()).or_insert_with(|| IndexBuild {
            unique,
            index_type: index_type.clone(),
            columns: Vec::new(),
        });
        if index.unique != unique || index.index_type != index_type {
            return Err(DbStateError::InconsistentIndexMetadata { table, index: name });
        }
        let sub_is_null = bool01(&row[6])?;
        index.columns.push(IndexColumnInventory {
            sequence: u32_value(&row[4])?,
            column_name,
            sub_part: if sub_is_null { None } else { Some(u32_value(&row[7])?) },
            descending,
        });
    }

    let fk_capture = query(plan, &foreign_keys_sql(&plan.database_name)?)?;
    probes.push(probe("03-foreign-keys", &fk_capture)?);
    for row in rows(&fk_capture.stdout, 6, true)? {
        let table = hex_text(&row[0])?;
        let name = hex_text(&row[1])?;
        let column = hex_text(&row[3])?;
        let referenced_table = hex_text(&row[4])?;
        let referenced_column = hex_text(&row[5])?;
        if name.is_empty() || column.is_empty() || referenced_table.is_empty() || referenced_column.is_empty() {
            return Err(DbStateError::UnexpectedQueryShape);
        }
        let target = tables.get_mut(&table).ok_or_else(|| DbStateError::UnknownTable(table.clone()))?;
        let fk = target.fks.entry(name.clone()).or_insert_with(|| ForeignKeyBuild {
            referenced_table: referenced_table.clone(),
            columns: Vec::new(),
        });
        if fk.referenced_table != referenced_table {
            return Err(DbStateError::InconsistentForeignKeyMetadata { table, constraint: name });
        }
        fk.columns.push((u32_value(&row[2])?, column, referenced_column));
    }

    let mut inventories = Vec::with_capacity(tables.len());
    for (table_name, mut build) in tables {
        let quoted = quote_identifier(&table_name)?;
        let ddl_capture = query(plan, &format!("SHOW CREATE TABLE {quoted}"))?;
        probes.push(probe(&format!("10-ddl:{table_name}"), &ddl_capture)?);
        let count_capture = query(plan, &format!("SELECT COUNT(*) FROM {quoted}"))?;
        probes.push(probe(&format!("11-count:{table_name}"), &count_capture)?);

        build.columns.sort_by_key(|column| column.ordinal_position);
        let indexes = build.indexes.into_iter().map(|(name, mut index)| {
            index.columns.sort_by_key(|column| column.sequence);
            IndexInventory { name, unique: index.unique, index_type: index.index_type, columns: index.columns }
        }).collect();
        let foreign_keys = build.fks.into_iter().map(|(name, mut fk)| {
            fk.columns.sort_by_key(|(ordinal, _, _)| *ordinal);
            let local = fk.columns.iter().map(|(_, column, _)| column.clone()).collect();
            let referenced = fk.columns.into_iter().map(|(_, _, column)| column).collect();
            ForeignKeyInventory {
                name,
                columns: local,
                referenced_table: fk.referenced_table,
                referenced_columns: referenced,
            }
        }).collect();
        inventories.push(TableInventory {
            name: table_name,
            engine: build.engine,
            show_create_table_sha256: sha256_hex(&ddl_capture.stdout),
            exact_row_count: single_u64(&count_capture.stdout)?,
            columns: build.columns,
            indexes,
            foreign_keys,
        });
    }

    let inventory = MySqlSchemaInventory {
        schema_version: 1,
        database_name: plan.database_name.clone(),
        tables: inventories,
    };
    inventory.validate().map_err(inv_err)?;
    let schema_sha = inventory.inventory_sha256().map_err(inv_err)?;
    let row_sha = inventory.row_count_projection_sha256().map_err(inv_err)?;
    let import_process_sha = import.evidence_sha256().map_err(exec_err)?;
    let import_log_sha = import.import_log_sha256().map_err(exec_err)?;

    let state_manifest = DatabaseStateManifest {
        schema_version: 1,
        plan_sha256: plan_sha,
        import_process_evidence_sha256: import_process_sha.clone(),
        post_import_preflight_sha256: post_preflight_sha,
        schema_inventory_sha256: schema_sha.clone(),
        row_count_inventory_sha256: row_sha.clone(),
        probes,
    };
    let state_sha = state_manifest.manifest_sha256()?;
    let import_receipt = bind_successful_import(
        protocol,
        profile,
        acquisition,
        &import_process_sha,
        &import_log_sha,
        &schema_sha,
        &row_sha,
        &state_sha,
    ).map_err(|e| DbStateError::ImportReceipt(e.to_string()))?;
    let schema_import_binding = bind_schema_inventory_to_import(&inventory, &import_receipt).map_err(inv_err)?;

    let state = VerifiedImportedDatabaseState {
        post_import_preflight,
        inventory,
        state_manifest,
        import_receipt,
        schema_import_binding,
    };
    state.evidence_sha256()?;
    Ok(state)
}

fn query(plan: &OqmdLocalImportPlan, sql: &str) -> Result<ProcessCapture, DbStateError> {
    if sql.chars().any(|ch| ch == '\0') {
        return Err(DbStateError::InvalidQuery);
    }
    let capture = capture_process(&ProcessSpec {
        command: plan.mysql_client.path.clone(),
        args: vec![
            "--no-defaults".to_string(),
            "--protocol=SOCKET".to_string(),
            format!("--socket={}", plan.unix_socket_path),
            format!("--user={}", plan.mysql_user),
            "--batch".to_string(),
            "--raw".to_string(),
            "--skip-column-names".to_string(),
            format!("--execute={sql}"),
            plan.database_name.clone(),
        ],
        environment: deterministic_environment(),
        environment_policy: EnvironmentPolicy::ClearAndSet,
        timeout_ms: plan.probe_timeout_ms,
        max_output_bytes: plan.max_output_bytes,
    }).map_err(|e| DbStateError::Process(e.to_string()))?;
    if !capture.process_success() {
        return Err(DbStateError::QueryDidNotSucceed(sql.to_string()));
    }
    if capture.stdout_truncated || capture.stderr_truncated {
        return Err(DbStateError::QueryOutputTruncated(sql.to_string()));
    }
    Ok(capture)
}

fn probe(label: &str, capture: &ProcessCapture) -> Result<DatabaseProbeDigest, DbStateError> {
    Ok(DatabaseProbeDigest {
        label: label.to_string(),
        command_manifest_sha256: capture.command_manifest_sha256.clone(),
        process_capture_sha256: capture.capture_sha256().map_err(|e| DbStateError::Process(e.to_string()))?,
        stdout_sha256: sha256_hex(&capture.stdout),
    })
}

fn tables_sql(db: &str) -> Result<String, DbStateError> {
    let db = sql_literal(db)?;
    Ok(format!("SELECT HEX(TABLE_NAME),HEX(COALESCE(ENGINE,'')) FROM information_schema.TABLES WHERE TABLE_SCHEMA='{db}' AND TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"))
}

fn columns_sql(db: &str) -> Result<String, DbStateError> {
    let db = sql_literal(db)?;
    Ok(format!("SELECT HEX(TABLE_NAME),ORDINAL_POSITION,HEX(COLUMN_NAME),HEX(COLUMN_TYPE),IF(IS_NULLABLE='YES',1,0),IF(COLUMN_DEFAULT IS NULL,1,0),HEX(COALESCE(COLUMN_DEFAULT,'')),IF(COLLATION_NAME IS NULL,1,0),HEX(COALESCE(COLLATION_NAME,'')),HEX(COALESCE(EXTRA,'')) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA='{db}' ORDER BY TABLE_NAME,ORDINAL_POSITION"))
}

fn indexes_sql(db: &str) -> Result<String, DbStateError> {
    let db = sql_literal(db)?;
    Ok(format!("SELECT HEX(TABLE_NAME),HEX(INDEX_NAME),NON_UNIQUE,HEX(INDEX_TYPE),SEQ_IN_INDEX,HEX(COALESCE(COLUMN_NAME,'')),IF(SUB_PART IS NULL,1,0),COALESCE(SUB_PART,0),HEX(COALESCE(COLLATION,'')) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA='{db}' ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX"))
}

fn foreign_keys_sql(db: &str) -> Result<String, DbStateError> {
    let db = sql_literal(db)?;
    Ok(format!("SELECT HEX(TABLE_NAME),HEX(CONSTRAINT_NAME),ORDINAL_POSITION,HEX(COLUMN_NAME),HEX(REFERENCED_TABLE_NAME),HEX(REFERENCED_COLUMN_NAME) FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA='{db}' AND REFERENCED_TABLE_NAME IS NOT NULL ORDER BY TABLE_NAME,CONSTRAINT_NAME,ORDINAL_POSITION"))
}

fn rows(bytes: &[u8], width: usize, allow_empty: bool) -> Result<Vec<Vec<String>>, DbStateError> {
    let text = std::str::from_utf8(bytes).map_err(|_| DbStateError::QueryOutputNotUtf8)?;
    let text = text.trim_end_matches(|ch| ch == '\r' || ch == '\n');
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split('\n').map(|line| {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let row: Vec<String> = line.split('\t').map(ToString::to_string).collect();
        if row.len() != width || (!allow_empty && row.iter().any(String::is_empty)) {
            Err(DbStateError::UnexpectedQueryShape)
        } else {
            Ok(row)
        }
    }).collect()
}

fn hex_text(value: &str) -> Result<String, DbStateError> {
    if value.len() % 2 != 0 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(DbStateError::InvalidHex);
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for chunk in value.as_bytes().chunks_exact(2) {
        let pair = std::str::from_utf8(chunk).map_err(|_| DbStateError::InvalidHex)?;
        decoded.push(u8::from_str_radix(pair, 16).map_err(|_| DbStateError::InvalidHex)?);
    }
    String::from_utf8(decoded).map_err(|_| DbStateError::DecodedMetadataNotUtf8)
}

fn bool01(value: &str) -> Result<bool, DbStateError> {
    match value { "0" => Ok(false), "1" => Ok(true), _ => Err(DbStateError::UnexpectedQueryShape) }
}

fn u32_value(value: &str) -> Result<u32, DbStateError> {
    value.parse().map_err(|_| DbStateError::UnexpectedQueryShape)
}

fn single_u64(bytes: &[u8]) -> Result<u64, DbStateError> {
    let text = std::str::from_utf8(bytes).map_err(|_| DbStateError::QueryOutputNotUtf8)?.trim();
    if text.is_empty() || text.chars().any(char::is_whitespace) {
        return Err(DbStateError::UnexpectedQueryShape);
    }
    text.parse().map_err(|_| DbStateError::UnexpectedQueryShape)
}

fn sql_literal(value: &str) -> Result<String, DbStateError> {
    if value.is_empty() || value.chars().any(|ch| matches!(ch, '\0' | '\n' | '\r' | '\'' | '\\')) {
        return Err(DbStateError::InvalidSqlLiteral);
    }
    Ok(value.to_string())
}

fn quote_identifier(value: &str) -> Result<String, DbStateError> {
    if value.is_empty() || value.chars().any(|ch| ch == '\0') {
        return Err(DbStateError::InvalidIdentifier);
    }
    Ok(format!("`{}`", value.replace('`', "``")))
}

fn deterministic_environment() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("LANG".to_string(), "C".to_string()),
        ("LC_ALL".to_string(), "C".to_string()),
        ("TZ".to_string(), "UTC".to_string()),
    ])
}

fn valid_sha(value: &str) -> Result<(), DbStateError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(DbStateError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn hash_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> { Ok(sha256_hex(&serde_json::to_vec(value)?)) }
fn inv_err<E: std::fmt::Display>(e: E) -> DbStateError { DbStateError::Inventory(e.to_string()) }
fn exec_err<E: std::fmt::Display>(e: E) -> DbStateError { DbStateError::ImportExecutor(e.to_string()) }

/// Canonical database-state capture failure.
#[derive(Debug, Error)]
pub enum DbStateError {
    #[error("import executor rejected input: {0}")] ImportExecutor(String),
    #[error("canonical inventory rejected input: {0}")] Inventory(String),
    #[error("import receipt rejected input: {0}")] ImportReceipt(String),
    #[error("raw process failure: {0}")] Process(String),
    #[error("unsupported database-state manifest schema {0}")] UnsupportedManifestSchema(u32),
    #[error("database-state manifest contains no probes")] EmptyProbeSet,
    #[error("invalid or duplicate probe label: {0}")] InvalidProbeLabel(String),
    #[error("mysql import evidence belongs to another plan")] ImportPlanMismatch,
    #[error("mysql import process did not succeed")] ImportProcessDidNotSucceed,
    #[error("mysql import diagnostic output was truncated")] ImportDiagnosticsTruncated,
    #[error("imported database contains no base tables")] EmptyDatabase,
    #[error("invalid or duplicate table metadata: {0}")] InvalidTableMetadata(String),
    #[error("metadata refers to unknown table: {0}")] UnknownTable(String),
    #[error("unsupported index shape: {table}.{index}")] UnsupportedIndexShape { table: String, index: String },
    #[error("unexpected index collation marker: {0}")] UnexpectedIndexCollation(String),
    #[error("inconsistent index metadata: {table}.{index}")] InconsistentIndexMetadata { table: String, index: String },
    #[error("inconsistent foreign-key metadata: {table}.{constraint}")] InconsistentForeignKeyMetadata { table: String, constraint: String },
    #[error("mysql query did not succeed: {0}")] QueryDidNotSucceed(String),
    #[error("mysql query output was truncated: {0}")] QueryOutputTruncated(String),
    #[error("invalid query")] InvalidQuery,
    #[error("query output is not UTF-8")] QueryOutputNotUtf8,
    #[error("query output has unexpected shape")] UnexpectedQueryShape,
    #[error("invalid hex-encoded metadata")] InvalidHex,
    #[error("decoded database metadata is not UTF-8")] DecodedMetadataNotUtf8,
    #[error("unsafe SQL literal")] InvalidSqlLiteral,
    #[error("invalid SQL identifier")] InvalidIdentifier,
    #[error("invalid SHA-256: {0}")] InvalidSha256(String),
    #[error(transparent)] Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullable_metadata_rows_can_contain_empty_hex_fields() {
        let parsed = rows(b"656E7472696573\t1\t6964\t626967696E74\t0\t1\t\t1\t\t\n", 10, true).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0][6].is_empty());
    }

    #[test]
    fn hex_decoder_and_count_parser_are_strict() {
        assert_eq!(hex_text("4665436F5A72").unwrap(), "FeCoZr");
        assert!(hex_text("xyz").is_err());
        assert_eq!(single_u64(b"42\n").unwrap(), 42);
        assert!(single_u64(b"42 43\n").is_err());
    }

    #[test]
    fn identifiers_are_backtick_escaped() {
        assert_eq!(quote_identifier("a`b").unwrap(), "`a``b`");
    }
}
