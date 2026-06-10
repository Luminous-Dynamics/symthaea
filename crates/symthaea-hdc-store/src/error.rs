// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for the HDC store.

use thiserror::Error;

/// Errors that can occur during HDC store operations.
#[derive(Debug, Error)]
pub enum HdcStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid store header: {reason}")]
    InvalidHeader { reason: String },
    #[error("Store is full (capacity: {capacity})")]
    StoreFull { capacity: u64 },
    #[error("Entry not found: id={id}")]
    NotFound { id: u64 },
    #[error("Duplicate entry: id={id}")]
    Duplicate { id: u64 },
    #[error("Compaction failed: {reason}")]
    CompactionFailed { reason: String },
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },
}
