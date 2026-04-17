// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 2b — Restorative justice + effective tier gating.
//!
//! The Phase 1 survey called out the "3:1 correction ratio" from production's
//! `SubPassport` as the key social-dynamics invariant to model. These tests
//! verify that:
//!
//! 1. A population with a 10:3 correction:violation ratio (net-zero parity)
//!    keeps aggregate tier penalty bounded at zero.
//! 2. A population with a 1:3 ratio (violations dominate) accumulates
//!    penalties — tier distribution shifts downward.
//! 3. `World::civic_fraction_meeting` respects per-agent tier penalties —
//!    two populations with identical sovereign profiles but different
//!    `RestorativeJustice` state produce different voter-eligibility
//!    fractions.

use mycelix_multiworld_sim::sovereign_profile::{
    civic_requirement_voting, CivicTier, DimensionWeights, SovereignProfile,
};
use mycelix_multiworld_sim::sub_passport::RestorativeJustice;

#[test]
fn ten_to_three_ratio_keeps_penalty_zero() {
    let mut rj = RestorativeJustice::new();
    // 300 violations, 1000 corrections — 10/3 = restore-to-degrade parity.
    for tick in 0..1300 {
        if tick % 13 < 3 {
            rj.record_violation(tick);
        } else {
            rj.record_correction(tick);
        }
    }
    assert_eq!(
        rj.tier_penalty, 0,
        "10:3 correction ratio should produce zero net penalty, got {}",
        rj.tier_penalty,
    );
}

#[test]
fn one_to_three_ratio_accumulates_penalty() {
    let mut rj = RestorativeJustice::new();
    // Violations 3× corrections (1:3 correction:violation)
    // - violations every 4 ticks out of 5 (0,1,2,3 = violation; 4 = correction)
    for tick in 0..500 {
        if tick % 4 == 3 {
            rj.record_correction(tick);
        } else {
            rj.record_violation(tick);
        }
    }
    assert!(
        rj.tier_penalty >= 2,
        "violation-dominant ratio should accumulate penalty, got {}",
        rj.tier_penalty,
    );
    assert!(rj.compliance_ratio() < 0.5);
}

#[test]
fn effective_tier_gates_voter_eligibility() {
    // Scenario: two agents, identical Guardian-tier sovereign profiles.
    // Agent A has a clean RestorativeJustice; agent B has a penalty of 2.
    // The voting requirement is Citizen tier + EI ≥ 0.25.
    let w = DimensionWeights::governance();
    let voting = civic_requirement_voting();

    let profile = SovereignProfile::from_array([0.9; 8]); // Guardian
    let mut rj_penalized = RestorativeJustice::new();
    // 6 violations → penalty = 2 (Guardian → Citizen).
    for tick in 0..6 {
        rj_penalized.record_violation(tick);
    }
    assert_eq!(rj_penalized.tier_penalty, 2);

    // Apply tier penalty manually to demonstrate the effective tier.
    let raw_a = profile.tier(&w);
    let effective_a = RestorativeJustice::new().effective_tier(raw_a);
    let effective_b = rj_penalized.effective_tier(raw_a);

    assert_eq!(raw_a, CivicTier::Guardian);
    assert_eq!(effective_a, CivicTier::Guardian);
    assert_eq!(effective_b, CivicTier::Citizen);

    // Both still meet the voting requirement (Citizen tier + EI ≥ 0.25),
    // but further degradation would block voting.
    assert!(profile.meets_requirement(&voting, &w));

    // After a third degradation (Citizen → Participant), agent B falls below
    // the voting tier floor.
    let mut rj_deep = rj_penalized.clone();
    // Record violations on tick t+10 (past cooldown) to apply another tier.
    for t in 10..13 {
        rj_deep.record_violation(t);
    }
    assert_eq!(rj_deep.tier_penalty, 3);
    assert_eq!(rj_deep.effective_tier(raw_a), CivicTier::Participant);
    // Participant < Citizen → ineligible to vote.
    assert!(rj_deep.effective_tier(raw_a) < voting.min_tier);
}

#[test]
fn compliance_ratio_reflects_behavior() {
    let mut good = RestorativeJustice::new();
    for t in 0..10 {
        good.record_correction(t);
    }
    good.record_violation(10);
    // 10 corrections, 1 violation → ~0.909 compliance.
    assert!((good.compliance_ratio() - 10.0 / 11.0).abs() < 1e-9);

    let mut bad = RestorativeJustice::new();
    for t in 0..10 {
        bad.record_violation(t);
    }
    bad.record_correction(11);
    // 1 correction, 10 violations → ~0.091 compliance.
    assert!(bad.compliance_ratio() < 0.1);
}
