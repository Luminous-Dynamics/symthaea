// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! symthaea-hdc-store -- Zero-copy mmap'd BinaryHV storage with snapshot-accelerated LSH.
//!
//! Provides efficient on-disk storage for 16,384-bit BinaryHV vectors with
//! O(1) random access via memory-mapped I/O and approximate nearest neighbor
//! search via a validated locality-sensitive hashing index with optional atomic snapshots.

pub mod batch;
mod checksum;
pub mod compaction;
pub mod content_checksum;
pub mod error;
mod fault;
pub mod header;
pub mod health;
pub mod inspection;
pub mod locking;
pub mod lsh_persistent;
pub mod lsh_snapshot;
pub mod migration;
pub mod portable_archive;
pub mod read_view;
pub mod reader;
pub mod recovery;
pub mod search;
pub mod store;
mod transaction;
pub mod validation;

pub use content_checksum::StoreContentChecksum;

pub use batch::{BatchCommitReport, BatchRecoveryDisposition, BatchRecoveryReport, WriteBatch};
pub use error::HdcStoreError;
pub use header::{HeaderSlot, StoreHeader};
pub use health::StoreHealth;
pub use inspection::{HeaderSlotInspection, InspectionIssue, StoreInspection, inspect_store};
pub use locking::store_lock_path;
pub use lsh_persistent::{DEFAULT_LSH_SEED, LshIndex, LshSignature};
pub use lsh_snapshot::{
    IndexLoadSource, IndexOpenPolicy, IndexStatus, LshSnapshot, LshSnapshotMetadata,
    load_lsh_snapshot, lsh_snapshot_path, write_lsh_snapshot,
};
pub use migration::{MigrationReport, migrate_v1};
pub use portable_archive::{
    PortableArchiveLimits, PortableArchiveMetadata, PortableExportReport, PortableRestoreReport,
    export_portable_archive, export_portable_archive_with_limits, inspect_portable_archive,
    inspect_portable_archive_with_limits, restore_portable_archive,
    restore_portable_archive_with_limits,
};
pub use read_view::HdcReadView;
pub use reader::HdcStoreReader;
pub use recovery::{HeaderHealth, RecoveryReport};
pub use search::{ApproximateSearchOptions, SearchOutcome};
pub use store::{HdcStore, StoreConfig};
pub use transaction::batch_journal_path;
pub use validation::{
    AnnGateResult, AnnQueryResult, AnnValidationReport, AnnValidationSuite, AnnValidationThresholds,
};
