// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent compile/test harness for WCARE adjudication infrastructure.
//!
//! Adjudication remains qualification infrastructure rather than live cognition.

mod evaluation_contract {
    pub use symthaea_wisdom::{GateClass, ScenarioFamily, ScenarioSpec, WCARE_V1_SCENARIOS};
}

mod qualification_receipt {
    pub use symthaea_wisdom::{ScenarioOutcome, WCARE_V1_CONTRACT_ID};
}

#[path = "../src/corpus_manifest.rs"]
mod corpus_manifest;

#[path = "../src/adjudication.rs"]
mod adjudication;

#[test]
fn adjudication_stays_outside_live_wisdom_authority() {
    assert!(evaluation_contract::WCARE_V1_SCENARIOS.len() >= 10);
    let policy = adjudication::AdjudicationPolicy::new(2, true)
        .expect("independent adjudication policy should be valid");
    assert_eq!(policy.minimum_independent_lineages, 2);
}
