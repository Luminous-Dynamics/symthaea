// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Journaled multi-record mutation batches.

use symthaea_core::hdc::BinaryHV;

/// A set of appends and deletions published by one header generation.
///
/// IDs may occur at most once across the complete batch. Appended IDs must not
/// already be live, and deleted IDs must exist when the batch is validated.
#[derive(Default)]
pub struct WriteBatch {
    pub(crate) appends: Vec<(u64, BinaryHV)>,
    pub(crate) deletes: Vec<u64>,
}

impl WriteBatch {
    /// Construct an empty batch.
    pub const fn new() -> Self {
        Self {
            appends: Vec::new(),
            deletes: Vec::new(),
        }
    }

    /// Add an append operation.
    pub fn push_append(&mut self, id: u64, hv: BinaryHV) {
        self.appends.push((id, hv));
    }

    /// Add a delete operation.
    pub fn push_delete(&mut self, id: u64) {
        self.deletes.push(id);
    }

    /// Builder-style append operation.
    pub fn append(mut self, id: u64, hv: BinaryHV) -> Self {
        self.push_append(id, hv);
        self
    }

    /// Builder-style delete operation.
    pub fn delete(mut self, id: u64) -> Self {
        self.push_delete(id);
        self
    }

    /// Number of operations in the batch.
    pub fn len(&self) -> usize {
        self.appends.len().saturating_add(self.deletes.len())
    }

    /// Whether the batch contains no operations.
    pub fn is_empty(&self) -> bool {
        self.appends.is_empty() && self.deletes.is_empty()
    }
}

/// Result of a successfully published write batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCommitReport {
    pub generation_before: u64,
    pub generation_after: u64,
    pub appended: u64,
    pub deleted: u64,
}

impl BatchCommitReport {
    /// Whether the batch changed the canonical store.
    pub const fn changed_store(&self) -> bool {
        self.appended != 0 || self.deleted != 0
    }
}

/// How a pending write journal was resolved during recovering open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchRecoveryDisposition {
    /// The publishing header never committed, so entry changes were reverted.
    RolledBack,
    /// The publishing header committed, so only stale journal cleanup remained.
    FinalizedCommitted,
}

/// Audit record for automatic recovery of one pending batch journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRecoveryReport {
    pub base_generation: u64,
    pub target_generation: u64,
    pub appended: u64,
    pub deleted: u64,
    pub disposition: BatchRecoveryDisposition,
}
