// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end test: run the `proposal_vote_invariant` scenario against the
//! in-repo `mycelix-mock-conductor` binary.
//!
//! This test is the M3 structural gate — it proves the scenario harness
//! works end-to-end without a live conductor. Live-conductor runs use the
//! same scenario with a different `bridge_binary`.
//!
//! # Requires feature flag
//!
//! The mock binary is gated behind `mock-conductor`. Run with:
//! ```bash
//! cargo test --features mock-conductor --test scenario_proposal_vote
//! ```

#![cfg(feature = "mock-conductor")]

use std::path::PathBuf;

use symtropy_mycelix_bridge::{proposal_vote_invariant, ScenarioConfig};

/// Locate the `mycelix-mock-conductor` binary built by this crate. Tests
/// run after `cargo build --features mock-conductor`, so the artifact is
/// in `target/debug/` (or `target/<profile>/`). We check both.
fn mock_binary_path() -> PathBuf {
    // $CARGO_TARGET_DIR/<profile>/mycelix-mock-conductor
    // Fall back to the test's own binary location.
    let current_exe = std::env::current_exe().expect("test binary path");
    // current_exe is e.g. target/debug/deps/scenario_proposal_vote-<hash>
    // The mock lives two dirs up: target/debug/mycelix-mock-conductor
    let mut path = current_exe.clone();
    path.pop(); // drop binary name
    path.pop(); // drop `deps`
    path.push("mycelix-mock-conductor");
    if path.exists() {
        return path;
    }
    panic!(
        "mock conductor binary not found at {}. Run `cargo build --features \
         mock-conductor --bin mycelix-mock-conductor` first.",
        path.display()
    );
}

#[test]
fn five_agent_proposal_vote_invariant_against_mock() {
    let config = ScenarioConfig::mock_default().with_bridge_binary(mock_binary_path());
    let report = proposal_vote_invariant(config);
    assert!(
        report.passed,
        "scenario failed: {} ({} ticks, {} errors)",
        report.summary, report.ticks, report.errors
    );
    assert_eq!(report.errors, 0);
    // 5 SubmitProposal + 5 GetProposal = 10 sent; 10 responses.
    assert_eq!(report.requests_sent, 10);
    assert_eq!(report.responses_received, 10);
}

#[test]
fn ten_agent_scale_test_against_mock() {
    let config = ScenarioConfig::mock_default()
        .with_agents(10)
        .with_tick_budget(120)
        .with_bridge_binary(mock_binary_path());
    let report = proposal_vote_invariant(config);
    assert!(
        report.passed,
        "10-agent scenario failed: {} ({} ticks)",
        report.summary, report.ticks
    );
    assert_eq!(report.requests_sent, 20); // 10 submits + 10 get_proposal
    assert_eq!(report.responses_received, 20);
}

#[test]
fn one_agent_degenerate_case() {
    let config = ScenarioConfig::mock_default()
        .with_agents(1)
        .with_bridge_binary(mock_binary_path());
    let report = proposal_vote_invariant(config);
    assert!(report.passed, "1-agent scenario failed: {}", report.summary);
    assert_eq!(report.requests_sent, 2); // 1 submit + 1 get_proposal
}
