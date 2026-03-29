// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compaction utilities for HdcStore.
//!
//! Provides automatic compaction triggering and statistics.

/// Statistics about compaction needs.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Current live count.
    pub live_count: u64,
    /// Current tombstone count.
    pub tombstone_count: u64,
    /// Tombstone ratio (tombstones / (live + tombstones)).
    pub tombstone_ratio: f64,
    /// Whether compaction is recommended.
    pub recommended: bool,
}

/// Compaction threshold: tombstones > 25% of live entries.
pub const COMPACTION_THRESHOLD: f64 = 0.25;

/// Compute compaction statistics.
pub fn compaction_stats(live: u64, tombstones: u64) -> CompactionStats {
    let total = live + tombstones;
    let ratio = if total > 0 {
        tombstones as f64 / total as f64
    } else {
        0.0
    };
    CompactionStats {
        live_count: live,
        tombstone_count: tombstones,
        tombstone_ratio: ratio,
        recommended: live > 0 && tombstones > (live as f64 * COMPACTION_THRESHOLD) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_no_entries() {
        let stats = compaction_stats(0, 0);
        assert!(!stats.recommended);
        assert_eq!(stats.tombstone_ratio, 0.0);
    }

    #[test]
    fn stats_no_tombstones() {
        let stats = compaction_stats(100, 0);
        assert!(!stats.recommended);
    }

    #[test]
    fn stats_below_threshold() {
        let stats = compaction_stats(100, 20);
        assert!(!stats.recommended);
    }

    #[test]
    fn stats_above_threshold() {
        let stats = compaction_stats(100, 30);
        assert!(stats.recommended);
    }
}
