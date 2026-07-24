// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit metadata recovery diagnostics for format-v2 stores.

use crate::batch::BatchRecoveryReport;
use crate::header::HeaderSlot;

/// Whether both independently checksummed header pages were valid at open time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderHealth {
    /// Both header pages were valid. They may have different generations, which
    /// is normal because commits alternate between slots.
    Redundant,
    /// Exactly one header page was valid.
    Degraded {
        /// Slot selected as the newest valid header.
        valid_slot: HeaderSlot,
        /// Slot that failed validation.
        invalid_slot: HeaderSlot,
        /// Validation error for the invalid slot.
        reason: String,
    },
}

impl HeaderHealth {
    /// Whether two valid header copies are currently available.
    pub const fn is_redundant(&self) -> bool {
        matches!(self, Self::Redundant)
    }
}

/// Audit trail returned by [`crate::HdcStore::open_recovering`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Slot selected before any repair writes.
    pub selected_slot: HeaderSlot,
    /// Generation selected before any repair writes.
    pub selected_generation: u64,
    /// Header redundancy observed before recovery.
    pub header_health_before: HeaderHealth,
    /// Whether live/tombstone counts were reconstructed from committed entries.
    pub repaired_entry_counts: bool,
    /// Whether recovery wrote a fresh copy into the alternate header page.
    pub repaired_header_redundancy: bool,
    /// Pending journal resolved before structural metadata recovery, if any.
    pub batch_recovery: Option<BatchRecoveryReport>,
    /// Number of contiguous live/tombstone-looking entries immediately after
    /// `vector_count`. They are reported but never promoted automatically.
    pub trailing_committed_entries: u64,
    /// Final committed generation after any repair.
    pub final_generation: u64,
}

impl RecoveryReport {
    /// Whether recovery changed on-disk metadata.
    pub const fn changed_store(&self) -> bool {
        self.batch_recovery.is_some()
            || self.repaired_entry_counts
            || self.repaired_header_redundancy
    }
}
