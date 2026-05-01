// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 2c — Mycelix-specific red-team scenarios.
//!
//! Survey Gap 5 called out 5 attack vectors unique to Mycelix's economic +
//! civic + restorative architecture:
//!
//! 1. TierBuyer — economic scale → artificial tier
//! 2. DemurrageEvader — stash-and-move to avoid SAP decay
//! 3. CorrectionFarmer — spam corrections to offset violations
//! 4. CrossClusterAmplifier — lenient gate bypass
//! 5. GuildColluder — coordinated peer-recognition
//!
//! This file proves each attack has a distinct modifier signature and that
//! the per-tick correction rate limit in `RestorativeJustice` actually
//! defeats `CorrectionFarmer`.

use mycelix_multiworld_sim::red_team::{
    evaluate_mycelix_resilience, AdversarialModifier, AdversarialStrategy,
};
use mycelix_multiworld_sim::sub_passport::{
    RestorativeJustice, CORRECTIONS_PER_RESTORE, MAX_CORRECTIONS_PER_TICK,
};

#[test]
fn all_five_mycelix_strategies_have_modifiers() {
    let strategies = [
        AdversarialStrategy::TierBuyer,
        AdversarialStrategy::DemurrageEvader,
        AdversarialStrategy::CorrectionFarmer,
        AdversarialStrategy::CrossClusterAmplifier,
        AdversarialStrategy::GuildColluder,
    ];
    for s in strategies {
        let m = AdversarialModifier::for_strategy(s, 0.01);
        assert_eq!(m.strategy, s, "strategy round-trip for {:?}", s);
    }
}

#[test]
fn correction_farming_attack_is_contained() {
    // Attack model: attacker alternates a violation with 10 corrections on
    // the same tick, repeatedly. Goal: keep tier_penalty at 0 indefinitely.
    let mut attacker = RestorativeJustice::new();

    for tick in 0..100u32 {
        // One violation.
        attacker.record_violation(tick);
        // Farmer spams 20 correction attempts per tick.
        for _ in 0..20 {
            attacker.record_correction(tick);
        }
    }

    // Baseline arithmetic: 100 violations → penalty would be 33 without cap;
    // clamped to MAX_TIER_PENALTY (4). Each tick credits only
    // `MAX_CORRECTIONS_PER_TICK = 2` corrections, so 200 total over 100 ticks.
    // 200 / CORRECTIONS_PER_RESTORE = 20 restore opportunities — BUT
    // restores are blocked on ticks where violations caused a degrade.
    assert!(
        attacker.tier_penalty > 0,
        "rate limit must prevent farmer from staying at zero penalty",
    );
    // And the farming-detection score should be very high.
    assert!(
        attacker.correction_farming_score() >= 0.9,
        "farming score too low: {}",
        attacker.correction_farming_score(),
    );
    // Credited corrections should be ≤ 2 × ticks.
    assert!(attacker.corrections <= MAX_CORRECTIONS_PER_TICK * 100);
}

#[test]
fn genuine_population_has_zero_farming_score() {
    // Normal population: one violation every 30 ticks, one correction every 10.
    let mut rj = RestorativeJustice::new();
    for tick in 0..3000u32 {
        if tick % 30 == 0 {
            rj.record_violation(tick);
        }
        if tick % 10 == 0 {
            rj.record_correction(tick);
        }
    }
    assert_eq!(rj.rejected_corrections, 0);
    assert_eq!(rj.correction_farming_score(), 0.0);
    // Correction-to-violation ratio is 3:1 (below the 10/3 net-zero ratio),
    // so a small positive penalty is expected — but it must stay bounded.
    assert!(rj.tier_penalty < 4, "penalty too high: {}", rj.tier_penalty);
}

#[test]
fn mycelix_resilience_summarizes_5_surfaces() {
    // Scenario: strong resilience on 4 surfaces, weak on correction-farming.
    let r = evaluate_mycelix_resilience(0.05, 0.10, 0.80, 0.15, 0.10);
    assert!(r.tier_buy_resilience > 0.9);
    assert!(r.demurrage_resilience > 0.85);
    assert!(r.correction_farm_resilience < 0.3);
    assert!(r.cross_cluster_resilience > 0.8);
    assert!(r.guild_collusion_resilience > 0.85);

    // Mean should be ~0.76 — strong overall but with a clear weak surface.
    let mean = r.mean();
    assert!(
        mean > 0.7 && mean < 0.85,
        "mean out of expected band: {}",
        mean
    );

    // 0.3 floor excludes correction-farming; we should flag no weak surface
    // above 0.3 but fail if we raise the floor.
    assert!(!r.no_weak_surface(0.5));
    assert!(!r.no_weak_surface(0.3));
}

#[test]
fn tier_buyer_and_demurrage_evader_are_distinguishable() {
    // Economic attacks target different surfaces.
    let tb = AdversarialModifier::for_strategy(AdversarialStrategy::TierBuyer, 0.05);
    let de = AdversarialModifier::for_strategy(AdversarialStrategy::DemurrageEvader, 0.05);

    // TierBuyer accumulates; DemurrageEvader churns.
    assert!(tb.sap_accumulation_mult > de.sap_accumulation_mult);
    assert!(de.sap_churn_mult > tb.sap_churn_mult);

    // TierBuyer looks legitimate; DemurrageEvader is detectable via churn.
    assert!(tb.appears_legitimate);
    assert!(!de.appears_legitimate);
}
