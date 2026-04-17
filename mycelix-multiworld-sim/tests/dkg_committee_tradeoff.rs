// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 3 — DKG committee-size tradeoff.
//!
//! The survey's framing for Phase 3 (Gap 3, Mycelix Sim): "Only needed if
//! calibrating how big governance committees can be before DKG round-trip
//! times make real-time decisions impractical."
//!
//! These tests answer three practical calibration questions:
//!
//! 1. At what committee size does the quadratic message cost dominate?
//! 2. At what dropout rate does a 50-member committee become non-viable?
//! 3. Is a 2/3 threshold or a 51% threshold more robust under 15% dropout?

use mycelix_multiworld_sim::dkg::{DkgConfig, DkgCost};

#[test]
fn quadratic_cost_dominates_beyond_50() {
    let small = DkgCost::estimate(&DkgConfig::new(7, 5, 1, 0.05));
    let medium = DkgCost::estimate(&DkgConfig::new(50, 34, 1, 0.05));
    let large = DkgCost::estimate(&DkgConfig::new(200, 134, 1, 0.05));

    // Message counts: 7×6×2=84, 50×49×2=4900, 200×199×2=79600.
    assert_eq!(small.message_count, 84);
    assert_eq!(medium.message_count, 4900);
    assert_eq!(large.message_count, 79_600);

    // Ratio: 200-member committee has ~950× more messages than 7-member.
    assert!(
        large.message_count as f64 / small.message_count as f64 > 500.0,
        "200-member committee cost not dominating: small={}, large={}",
        small.message_count,
        large.message_count,
    );
}

#[test]
fn dropout_collapses_viability_past_threshold() {
    // 50-member committee, 34/50 threshold. Sweep dropout.
    let sizes = [50u32];
    let threshold = 34u32;
    for &n in &sizes {
        for &dropout in &[0.0, 0.05, 0.10, 0.15, 0.25, 0.40] {
            let cfg = DkgConfig::new(n, threshold, 1, dropout);
            let cost = DkgCost::estimate(&cfg);
            if dropout <= 0.10 {
                assert!(
                    cost.success_prob > 0.90,
                    "50-of-{} should be robust at {}% dropout: {}",
                    threshold,
                    dropout * 100.0,
                    cost.success_prob,
                );
            }
            if dropout >= 0.25 {
                assert!(
                    cost.success_prob < 0.50,
                    "50-of-{} should collapse at {}% dropout: {}",
                    threshold,
                    dropout * 100.0,
                    cost.success_prob,
                );
            }
        }
    }
}

#[test]
fn majority_threshold_more_robust_than_supermajority() {
    // 50-member committee, 15% dropout. 2/3 threshold (34) vs 51% threshold (26).
    // The majority threshold should tolerate more dropout.
    let supermajority = DkgCost::estimate(&DkgConfig::new(50, 34, 1, 0.15));
    let majority = DkgCost::estimate(&DkgConfig::new(50, 26, 1, 0.15));

    assert!(
        majority.success_prob > supermajority.success_prob,
        "majority should be more robust: super={}, maj={}",
        supermajority.success_prob,
        majority.success_prob,
    );
    // Majority should clear a 90% reliability floor under 15% dropout.
    assert!(majority.is_viable(0.90));
}

#[test]
fn sim_tick_latency_bounds_committee_size() {
    // Governance uses 1 sim tick = 1 month. A round-trip across a real
    // Holochain committee takes seconds, so rtt_ticks = 0 (intra-tick)
    // for most realistic committee sizes. This test just documents that
    // the latency field is intended for scenarios where the ceremony
    // genuinely spans ticks (e.g., async deliberation / multi-day
    // threshold signing in slow-governance designs).
    let fast = DkgCost::estimate(&DkgConfig::new(50, 34, 0, 0.05));
    assert_eq!(fast.latency_ticks, 0);

    let slow = DkgCost::estimate(&DkgConfig::new(50, 34, 3, 0.05));
    assert_eq!(slow.latency_ticks, 6);
}
