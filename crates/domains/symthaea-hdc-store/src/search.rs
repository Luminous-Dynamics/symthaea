// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search contracts and diagnostics.

/// Controls when an approximate LSH query falls back to an exact scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApproximateSearchOptions {
    /// Fall back when LSH yields fewer than `top_k * candidate_multiplier`
    /// candidates. Set to zero to disable candidate-count fallback.
    pub candidate_multiplier: usize,
    /// Fall back to exact search when LSH yields no candidates.
    pub fallback_on_empty: bool,
}

impl Default for ApproximateSearchOptions {
    fn default() -> Self {
        Self {
            candidate_multiplier: 4,
            fallback_on_empty: true,
        }
    }
}

/// Search results plus enough diagnostics to audit approximate behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOutcome {
    /// Neighbors sorted by descending similarity and then ascending ID.
    pub neighbors: Vec<(u64, f32)>,
    /// Number of live vectors whose similarity was actually evaluated.
    pub examined: usize,
    /// Number of live vectors in the store when the search ran.
    pub total_live: usize,
    /// Whether every live vector was evaluated.
    pub exact: bool,
    /// Whether the configured candidate policy triggered a brute-force fallback.
    pub fell_back_to_exact: bool,
}
