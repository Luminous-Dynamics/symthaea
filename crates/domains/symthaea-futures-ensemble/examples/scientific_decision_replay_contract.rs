// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smoke entry point for the scientific-decision replay contract.
//! Test with: cargo test -p symthaea-futures-ensemble --example scientific_decision_replay_contract

#[path = "support/scientific_decision_replay.rs"]
#[allow(dead_code)]
mod scientific_decision_replay;

fn main() {
    println!(
        "Scientific decision replay schema v{} loaded; evaluator outcomes remain separate.",
        scientific_decision_replay::SCHEMA_VERSION
    );
}
