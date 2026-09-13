// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent compile/test harness for precommitted WCARE thresholds.

mod evaluation_contract {
    pub use symthaea_wisdom::{GateClass, ScenarioSpec, WCARE_V1_SCENARIOS};
}

#[path = "../src/threshold_policy.rs"]
mod threshold_policy;

#[test]
fn thresholds_remain_qualification_infrastructure() {
    assert!(evaluation_contract::WCARE_V1_SCENARIOS
        .iter()
        .any(|scenario| scenario.gate == evaluation_contract::GateClass::Comparative));
}
