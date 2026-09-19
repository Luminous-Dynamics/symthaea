// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical MySQL schema and exact row-count inventory for historical imports.
//!
//! An import receipt should not bind an opaque `schema_inventory_sha256` without a
//! deterministic object model. This crate makes the imported database state auditable
//! before extraction by freezing table definitions, columns, indexes, foreign keys,
//! and exact table row counts in canonical order.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_materials_historical_import::HistoricalImportExecutionReceipt;
use thiserror::Error;

/// One canonical column definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnInventory {
    /// 1-based ordinal position.
    pub ordinal_position: u32,
    /// Column name.
    pub name: String,
    /// Exact database-reported column type text.
    pub column_type: String,
    /// Whether NULL values are accepted.
    pub nullable: bool,
    /// Canonical database-reported default representation when present.
    pub default_repr: Option<String>,
    /// Character collation when relevant.
    pub collation: Option<String>,
    /// Exact database-reported extra flags.
    pub extra: String,
}

/// One ordered index column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexColumnInventory {
    /// 1-based index-column sequence.
    pub sequence: u32,
    /// Column name.
    pub column_name: String,
    /// Prefix length for prefix indexes.
    pub sub_part: Option<u32>,
    /// Whether descending ordering was reported.
    pub descending: bool,
}

/// One index definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexInventory {
    /// Index name.
    pub name: String,
    /// Whether duplicate keys are disallowed.
    pub unique: bool,
    /// Database-reported index method/type.
    pub index_type: String,
    /// Ordered index columns.
    pub columns: Vec<IndexColumnInventory>,
}

/// One foreign-key definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForeignKeyInventory {
    /// Constraint name.
    pub name: String,
    /// Ordered local columns.
    pub columns: Vec<String>,
    /// Referenced table name.
    pub referenced_table: String,
    /// Ordered referenced columns.
    pub referenced_columns: Vec<String>,
}

/// Canonical state for one imported table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableInventory {
    /// Table name.
    pub name: String,
    /// Storage engine reported by MySQL.
    pub engine: String,
    /// SHA-256 of canonical raw `SHOW CREATE TABLE` evidence.
    pub show_create_table_sha256: String,
    /// Exact `COUNT(*)` result captured after import.
    pub exact_row_count: u64,
    /// Columns ordered strictly by ordinal position.
    pub columns: Vec<ColumnInventory>,
    /// Indexes ordered lexically by name.
    pub indexes: Vec<IndexInventory>,
    /// Foreign keys ordered lexically by constraint name.
    pub foreign_keys: Vec<ForeignKeyInventory>,
}

/// Canonical imported database inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MySqlSchemaInventory {
    /// Inventory schema version.
    pub schema_version: u32,
    /// Imported database/schema name.
    pub database_name: String,
    /// Tables ordered lexically by table name.
    pub tables: Vec<TableInventory>,
}

impl MySqlSchemaInventory {
    /// Validate canonical ordering and structural invariants.
    pub fn validate(&self) -> Result<(), SchemaInventoryError> {
        if self.schema_version != 1 {
            return Err(SchemaInventoryError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        nonempty("database_name", &self.database_name)?;
        if self.tables.is_empty() {
            return Err(SchemaInventoryError::EmptyInventory);
        }

        let mut previous_table: Option<&str> = None;
        for table in &self.tables {
            table.validate()?;
            if previous_table.is_some_and(|prior| table.name.as_str() <= prior) {
                return Err(SchemaInventoryError::NonCanonicalTableOrder);
            }
            previous_table = Some(&table.name);
        }
        Ok(())
    }

    /// Deterministic SHA-256 of the full canonical schema inventory.
    pub fn inventory_sha256(&self) -> Result<String, SchemaInventoryError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Deterministic SHA-256 of only `(table_name, exact_row_count)` pairs.
    pub fn row_count_projection_sha256(&self) -> Result<String, SchemaInventoryError> {
        self.validate()?;
        let projection: Vec<(&str, u64)> = self
            .tables
            .iter()
            .map(|table| (table.name.as_str(), table.exact_row_count))
            .collect();
        Ok(sha256_hex(&serde_json::to_vec(&projection)?))
    }
}

impl TableInventory {
    fn validate(&self) -> Result<(), SchemaInventoryError> {
        nonempty("table_name", &self.name)?;
        nonempty("table_engine", &self.engine)?;
        sha256(&self.show_create_table_sha256)?;
        if self.columns.is_empty() {
            return Err(SchemaInventoryError::TableHasNoColumns(self.name.clone()));
        }

        for (index, column) in self.columns.iter().enumerate() {
            let expected = index as u32 + 1;
            if column.ordinal_position != expected {
                return Err(SchemaInventoryError::NonCanonicalColumnOrder {
                    table: self.name.clone(),
                });
            }
            nonempty("column_name", &column.name)?;
            nonempty("column_type", &column.column_type)?;
            if let Some(collation) = &column.collation {
                nonempty("column_collation", collation)?;
            }
        }

        let mut previous_index: Option<&str> = None;
        for index in &self.indexes {
            index.validate(&self.name)?;
            if previous_index.is_some_and(|prior| index.name.as_str() <= prior) {
                return Err(SchemaInventoryError::NonCanonicalIndexOrder {
                    table: self.name.clone(),
                });
            }
            previous_index = Some(&index.name);
        }

        let mut previous_fk: Option<&str> = None;
        for foreign_key in &self.foreign_keys {
            foreign_key.validate(&self.name)?;
            if previous_fk.is_some_and(|prior| foreign_key.name.as_str() <= prior) {
                return Err(SchemaInventoryError::NonCanonicalForeignKeyOrder {
                    table: self.name.clone(),
                });
            }
            previous_fk = Some(&foreign_key.name);
        }
        Ok(())
    }
}

impl IndexInventory {
    fn validate(&self, table: &str) -> Result<(), SchemaInventoryError> {
        nonempty("index_name", &self.name)?;
        nonempty("index_type", &self.index_type)?;
        if self.columns.is_empty() {
            return Err(SchemaInventoryError::IndexHasNoColumns {
                table: table.to_string(),
                index: self.name.clone(),
            });
        }
        for (index, column) in self.columns.iter().enumerate() {
            if column.sequence != index as u32 + 1 {
                return Err(SchemaInventoryError::NonCanonicalIndexColumnOrder {
                    table: table.to_string(),
                    index: self.name.clone(),
                });
            }
            nonempty("index_column_name", &column.column_name)?;
            if column.sub_part == Some(0) {
                return Err(SchemaInventoryError::InvalidIndexPrefix {
                    table: table.to_string(),
                    index: self.name.clone(),
                });
            }
        }
        Ok(())
    }
}

impl ForeignKeyInventory {
    fn validate(&self, table: &str) -> Result<(), SchemaInventoryError> {
        nonempty("foreign_key_name", &self.name)?;
        nonempty("referenced_table", &self.referenced_table)?;
        if self.columns.is_empty() || self.columns.len() != self.referenced_columns.len() {
            return Err(SchemaInventoryError::InvalidForeignKeyArity {
                table: table.to_string(),
                constraint: self.name.clone(),
            });
        }
        for column in &self.columns {
            nonempty("foreign_key_column", column)?;
        }
        for column in &self.referenced_columns {
            nonempty("referenced_column", column)?;
        }
        Ok(())
    }
}

/// Verified binding from a canonical inventory into an import receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaImportBinding {
    /// Full canonical schema inventory digest.
    pub schema_inventory_sha256: String,
    /// Canonical exact row-count projection digest.
    pub row_count_inventory_sha256: String,
    /// Import profile digest named by the receipt.
    pub import_profile_sha256: String,
}

impl SchemaImportBinding {
    /// Deterministic digest of this cross-object binding.
    pub fn binding_sha256(&self) -> Result<String, SchemaInventoryError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Require an import receipt to refer to the supplied canonical inventory bytes.
pub fn bind_schema_inventory_to_import(
    inventory: &MySqlSchemaInventory,
    import: &HistoricalImportExecutionReceipt,
) -> Result<SchemaImportBinding, SchemaInventoryError> {
    let schema_sha = inventory.inventory_sha256()?;
    let row_sha = inventory.row_count_projection_sha256()?;
    if !schema_sha.eq_ignore_ascii_case(&import.schema_inventory_sha256) {
        return Err(SchemaInventoryError::ImportSchemaDigestMismatch);
    }
    if !row_sha.eq_ignore_ascii_case(&import.row_count_inventory_sha256) {
        return Err(SchemaInventoryError::ImportRowCountDigestMismatch);
    }
    Ok(SchemaImportBinding {
        schema_inventory_sha256: schema_sha,
        row_count_inventory_sha256: row_sha,
        import_profile_sha256: import.profile_sha256.clone(),
    })
}

fn nonempty(name: &'static str, value: &str) -> Result<(), SchemaInventoryError> {
    if value.trim().is_empty() {
        return Err(SchemaInventoryError::EmptyField(name));
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), SchemaInventoryError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SchemaInventoryError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Canonical schema inventory failure.
#[derive(Debug, Error)]
pub enum SchemaInventoryError {
    /// Inventory schema unsupported.
    #[error("unsupported schema-inventory version {0}")]
    UnsupportedSchemaVersion(u32),
    /// Required field empty.
    #[error("required field {0} is empty")]
    EmptyField(&'static str),
    /// Inventory contains no tables.
    #[error("schema inventory contains no tables")]
    EmptyInventory,
    /// SHA malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Table order is not strict lexical order.
    #[error("tables are not in strict canonical lexical order")]
    NonCanonicalTableOrder,
    /// Table contains no columns.
    #[error("table {0} contains no columns")]
    TableHasNoColumns(String),
    /// Column ordinal ordering is not contiguous/canonical.
    #[error("table {table} columns are not in contiguous ordinal order")]
    NonCanonicalColumnOrder {
        /// Table name.
        table: String,
    },
    /// Index order is not canonical.
    #[error("table {table} indexes are not in strict lexical order")]
    NonCanonicalIndexOrder {
        /// Table name.
        table: String,
    },
    /// Index contains no columns.
    #[error("index {index} on table {table} has no columns")]
    IndexHasNoColumns {
        /// Table name.
        table: String,
        /// Index name.
        index: String,
    },
    /// Index-column sequence is not contiguous/canonical.
    #[error("index {index} on table {table} has noncanonical column order")]
    NonCanonicalIndexColumnOrder {
        /// Table name.
        table: String,
        /// Index name.
        index: String,
    },
    /// Index prefix length is invalid.
    #[error("index {index} on table {table} has invalid zero prefix length")]
    InvalidIndexPrefix {
        /// Table name.
        table: String,
        /// Index name.
        index: String,
    },
    /// Foreign-key order is not canonical.
    #[error("table {table} foreign keys are not in strict lexical order")]
    NonCanonicalForeignKeyOrder {
        /// Table name.
        table: String,
    },
    /// Foreign-key arity invalid.
    #[error("foreign key {constraint} on table {table} has invalid column arity")]
    InvalidForeignKeyArity {
        /// Table name.
        table: String,
        /// Constraint name.
        constraint: String,
    },
    /// Import receipt names another full schema inventory.
    #[error("import receipt schema inventory digest differs from canonical inventory")]
    ImportSchemaDigestMismatch,
    /// Import receipt names another exact row-count projection.
    #[error("import receipt row-count digest differs from canonical inventory")]
    ImportRowCountDigestMismatch,
    /// Serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn inventory() -> MySqlSchemaInventory {
        MySqlSchemaInventory {
            schema_version: 1,
            database_name: "oqmd_v17".to_string(),
            tables: vec![TableInventory {
                name: "entries".to_string(),
                engine: "InnoDB".to_string(),
                show_create_table_sha256: hex('a'),
                exact_row_count: 42,
                columns: vec![
                    ColumnInventory {
                        ordinal_position: 1,
                        name: "id".to_string(),
                        column_type: "bigint".to_string(),
                        nullable: false,
                        default_repr: None,
                        collation: None,
                        extra: "".to_string(),
                    },
                    ColumnInventory {
                        ordinal_position: 2,
                        name: "name".to_string(),
                        column_type: "varchar(255)".to_string(),
                        nullable: false,
                        default_repr: None,
                        collation: Some("utf8mb4_bin".to_string()),
                        extra: "".to_string(),
                    },
                ],
                indexes: vec![IndexInventory {
                    name: "PRIMARY".to_string(),
                    unique: true,
                    index_type: "BTREE".to_string(),
                    columns: vec![IndexColumnInventory {
                        sequence: 1,
                        column_name: "id".to_string(),
                        sub_part: None,
                        descending: false,
                    }],
                }],
                foreign_keys: Vec::new(),
            }],
        }
    }

    fn import_receipt(inventory: &MySqlSchemaInventory) -> HistoricalImportExecutionReceipt {
        HistoricalImportExecutionReceipt {
            schema_version: 1,
            profile_sha256: hex('1'),
            acquisition_receipt_sha256: hex('2'),
            compressed_snapshot_sha256: hex('3'),
            import_process_evidence_sha256: hex('4'),
            process_exit_success: true,
            import_log_sha256: hex('5'),
            schema_inventory_sha256: inventory.inventory_sha256().unwrap(),
            row_count_inventory_sha256: inventory.row_count_projection_sha256().unwrap(),
            database_state_manifest_sha256: hex('6'),
        }
    }

    #[test]
    fn canonical_inventory_binds_import_receipt() {
        let inventory = inventory();
        let import = import_receipt(&inventory);
        let binding = bind_schema_inventory_to_import(&inventory, &import).unwrap();
        assert_eq!(
            binding.schema_inventory_sha256,
            import.schema_inventory_sha256
        );
        assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    }

    #[test]
    fn changed_row_count_changes_both_full_and_projection_identity() {
        let original = inventory();
        let mut changed = original.clone();
        changed.tables[0].exact_row_count += 1;
        assert_ne!(
            original.inventory_sha256().unwrap(),
            changed.inventory_sha256().unwrap()
        );
        assert_ne!(
            original.row_count_projection_sha256().unwrap(),
            changed.row_count_projection_sha256().unwrap()
        );
    }

    #[test]
    fn reordered_tables_fail_closed() {
        let mut inventory = inventory();
        let mut second = inventory.tables[0].clone();
        second.name = "calculations".to_string();
        inventory.tables.push(second);
        assert!(matches!(
            inventory.validate(),
            Err(SchemaInventoryError::NonCanonicalTableOrder)
        ));
    }

    #[test]
    fn import_cannot_substitute_row_count_inventory() {
        let inventory = inventory();
        let mut import = import_receipt(&inventory);
        import.row_count_inventory_sha256 = hex('f');
        assert!(matches!(
            bind_schema_inventory_to_import(&inventory, &import),
            Err(SchemaInventoryError::ImportRowCountDigestMismatch)
        ));
    }
}
