// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Retired historical evidence surface.
//!
//! This target is intentionally retained so old invocations fail explicitly
//! instead of disappearing or continuing to emit evidence-looking numbers.
//! A replacement must establish a reachable live execution boundary and
//! explicit direction-aware metric policies under a separately qualified path.

const RETIREMENT_MARKER: &str = "RETIRED_EVIDENCE_SURFACE";
const REASON_EXECUTION: &str = "reason=no_reachable_live_backend_boundary";
const REASON_COMPARISON: &str = "reason=legacy_direction_blind_regression_comparator";
const REPLACEMENT: &str =
    "replacement=qualified_live_execution_receipt_plus_explicit_metric_policy_manifest";

fn main() {
    eprintln!("{RETIREMENT_MARKER}");
    eprintln!("{REASON_EXECUTION}");
    eprintln!("{REASON_COMPARISON}");
    eprintln!("{REPLACEMENT}");
    std::process::exit(2);
}
