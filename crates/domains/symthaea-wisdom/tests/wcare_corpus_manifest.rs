// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent compile/test harness for the WCARE corpus manifest.
//!
//! The manifest is qualification tooling, not runtime cognition. Keep it out of
//! the live Wisdom public API while still compiling and executing its invariants.

mod evaluation_contract {
    pub use symthaea_wisdom::{ScenarioFamily, ScenarioSpec, WCARE_V1_SCENARIOS};
}

mod qualification_receipt {
    pub use symthaea_wisdom::WCARE_V1_CONTRACT_ID;
}

#[path = "../src/corpus_manifest.rs"]
mod corpus_manifest;

#[test]
fn corpus_manifest_stays_measurement_only() {
    // If this harness compiles, the manifest can consume the frozen public
    // evaluation contract without becoming part of runtime Wisdom cognition.
    assert!(!evaluation_contract::WCARE_V1_SCENARIOS.is_empty());
}
