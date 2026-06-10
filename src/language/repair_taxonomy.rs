// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stable repair categories and hints for coding-agent feedback loops.

pub const FORCE_REPAIR_BENCH_ENV: &str = "SYMTHAEA_ENABLE_FORCED_REPAIR_BENCH";
pub const FORCE_GEODESIC_REJECTION_PREFIX: &str = "benchmark_force_geodesic_rejection:";
pub const FORCE_GEODESIC_REJECTION_UNLESS_REPAIR_MEMORY_PREFIX: &str =
    "benchmark_force_geodesic_rejection_unless_repair_memory:";

pub use symthaea_coding_feedback::{
    categorize_rejection, extract_embedded_category, extract_embedded_repair_hint,
    repair_lesson_for_rejection,
};

pub use symthaea_coding_feedback::repair_hint_for_rejection_category as repair_hint_for_category;

pub fn forced_geodesic_rejection(constraints: &[String]) -> Option<&str> {
    if std::env::var_os(FORCE_REPAIR_BENCH_ENV).is_none() {
        return None;
    }
    constraints
        .iter()
        .find_map(|constraint| constraint.strip_prefix(FORCE_GEODESIC_REJECTION_PREFIX))
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
}

pub fn forced_geodesic_rejection_unless_repair_memory(constraints: &[String]) -> Option<&str> {
    if std::env::var_os(FORCE_REPAIR_BENCH_ENV).is_none() {
        return None;
    }
    constraints
        .iter()
        .find_map(|constraint| {
            constraint.strip_prefix(FORCE_GEODESIC_REJECTION_UNLESS_REPAIR_MEMORY_PREFIX)
        })
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
}
