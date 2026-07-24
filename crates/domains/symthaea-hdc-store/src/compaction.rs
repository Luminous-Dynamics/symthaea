// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compaction policy and statistics for HdcStore.

/// Statistics about compaction needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionStats {
    /// Current live count.
    pub live_count: u64,
    /// Current tombstone count.
    pub tombstone_count: u64,
    /// Tombstones divided by all committed entries.
    pub tombstone_ratio: f64,
    /// Whether compaction is recommended by the shared policy.
    pub recommended: bool,
}

/// Recommend compaction when at least 25% of committed entries are tombstones.
pub const COMPACTION_TOMBSTONE_RATIO_THRESHOLD: f64 = 0.25;

/// Backward-compatible name for the compaction ratio threshold.
pub const COMPACTION_THRESHOLD: f64 = COMPACTION_TOMBSTONE_RATIO_THRESHOLD;

/// Compute compaction statistics using one unambiguous ratio definition.
pub fn compaction_stats(live: u64, tombstones: u64) -> CompactionStats {
    let total = live.saturating_add(tombstones);
    let ratio = if total > 0 {
        tombstones as f64 / total as f64
    } else {
        0.0
    };
    CompactionStats {
        live_count: live,
        tombstone_count: tombstones,
        tombstone_ratio: ratio,
        recommended: total > 0 && ratio >= COMPACTION_TOMBSTONE_RATIO_THRESHOLD,
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
        assert!(!compaction_stats(100, 0).recommended);
    }

    #[test]
    fn stats_below_threshold() {
        let stats = compaction_stats(100, 30);
        assert!(!stats.recommended);
        assert!(stats.tombstone_ratio < COMPACTION_TOMBSTONE_RATIO_THRESHOLD);
    }

    #[test]
    fn stats_at_threshold() {
        let stats = compaction_stats(75, 25);
        assert!(stats.recommended);
        assert_eq!(stats.tombstone_ratio, 0.25);
    }

    #[test]
    fn all_tombstones_recommends_compaction() {
        assert!(compaction_stats(0, 10).recommended);
    }
}
