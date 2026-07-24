// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for the HDC store.

use std::path::PathBuf;

use thiserror::Error;

/// Errors that can occur during HDC store operations.
#[derive(Debug, Error)]
pub enum HdcStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid store header: {reason}")]
    InvalidHeader { reason: String },
    #[error("Invalid store configuration: {reason}")]
    InvalidConfig { reason: String },
    #[error(
        "Header checksum mismatch at generation {generation}: expected {expected:#018x}, found {found:#018x}"
    )]
    HeaderChecksumMismatch {
        generation: u64,
        expected: u64,
        found: u64,
    },
    #[error("No valid redundant header was found: primary={primary}; secondary={secondary}")]
    NoValidHeader { primary: String, secondary: String },
    #[error("Conflicting headers have the same generation {generation}")]
    HeaderConflict { generation: u64 },
    #[error("Corrupt entry at index {index}: {reason}")]
    CorruptEntry { index: u64, reason: String },
    #[error("Store is already open for writing: {path:?}")]
    StoreLocked { path: PathBuf },
    #[error(
        "Store handle is poisoned after {operation}: {cause}; drop and reopen it before writing"
    )]
    StorePoisoned {
        operation: &'static str,
        cause: String,
    },
    #[error("Invalid LSH index snapshot at {path:?}: {reason}")]
    InvalidIndexSnapshot { path: PathBuf, reason: String },
    #[error("Invalid portable archive at {path:?}: {reason}")]
    InvalidPortableArchive { path: PathBuf, reason: String },
    #[error("Archive or restore destination already exists: {path:?}")]
    ArchiveDestinationExists { path: PathBuf },
    #[error("Portable archive resource limit exceeded at {path:?}: {reason}")]
    ArchiveLimitExceeded { path: PathBuf, reason: String },
    #[error("Invalid write batch: {reason}")]
    InvalidBatch { reason: String },
    #[error("A pending write batch journal requires recovering open: {path:?}")]
    PendingBatchTransaction { path: PathBuf },
    #[error("Invalid write batch journal at {path:?}: {reason}")]
    InvalidBatchJournal { path: PathBuf, reason: String },
    #[error("Integer overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
    #[error("Store is full (capacity: {capacity})")]
    StoreFull { capacity: u64 },
    #[error("Entry not found: id={id}")]
    NotFound { id: u64 },
    #[error("Duplicate entry: id={id}")]
    Duplicate { id: u64 },
    #[error("Compaction failed: {reason}")]
    CompactionFailed { reason: String },
    #[error("Migration failed: {reason}")]
    MigrationFailed { reason: String },
    #[error("Store already uses format version {version}; migration is not required")]
    MigrationNotRequired { version: u32 },
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },
}
