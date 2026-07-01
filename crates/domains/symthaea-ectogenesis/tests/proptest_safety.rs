// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based safety tests for ectogenesis crate.
//!
//! These tests verify safety-critical invariants that must hold for ALL inputs,
//! not just the specific scenarios covered by unit tests.

use proptest::prelude::*;
use symthaea_ectogenesis::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // ── Consent Proxy Monotonicity ──────────────────────────────────────────
    // ConsentProxy tier must never decrease with gestational age.
    // A week-24 fetus cannot regress to PreSentient.
    #[test]
    fn prop_consent_proxy_monotonic(w1 in 0u32..=40, w2 in 0u32..=40) {
        let tier1 = ConsentProxy::from_week(GestationalWeek::new(w1.min(w2)));
        let tier2 = ConsentProxy::from_week(GestationalWeek::new(w1.max(w2)));
        let rank = |t: &ConsentProxy| match t {
            ConsentProxy::PreSentient => 0,
            ConsentProxy::EmergingSentience => 1,
            ConsentProxy::Sentient => 2,
        };
        prop_assert!(rank(&tier2) >= rank(&tier1),
            "ConsentProxy must be monotonic: week {} -> {:?}, week {} -> {:?}",
            w1.min(w2), tier1, w1.max(w2), tier2);
    }

    // ── Minimum Benefit Monotonicity ────────────────────────────────────────
    // Required benefit threshold must never decrease with gestational age.
    #[test]
    fn prop_minimum_benefit_monotonic(w1 in 0u32..=40, w2 in 0u32..=40) {
        let gate = EctogenesisEthicsGate::fully_approved();
        let b1 = gate.minimum_benefit(GestationalWeek::new(w1.min(w2)));
        let b2 = gate.minimum_benefit(GestationalWeek::new(w1.max(w2)));
        prop_assert!(b2 >= b1,
            "minimum_benefit must be monotonic: week {} -> {}, week {} -> {}",
            w1.min(w2), b1, w1.max(w2), b2);
    }

    // ── Unapproved Gate Always Blocks Sentient Interventions ────────────────
    // No matter the benefit score, an unapproved gate must NEVER permit
    // interventions on a sentient fetus (week >= 24).
    #[test]
    fn prop_unapproved_blocks_sentient(week in 24u32..=42, benefit in 0.0f64..=1.0) {
        let gate = EctogenesisEthicsGate::unapproved();
        let result = gate.check_intervention(
            GestationalWeek::new(week),
            "test intervention",
            benefit,
        );
        prop_assert!(result.is_err(),
            "Unapproved gate must ALWAYS block sentient intervention at week {}, benefit={}",
            week, benefit);
    }

    // ── BioBag Quality Score Bounded ────────────────────────────────────────
    // quality_score() must always return [0.0, 1.0] regardless of fluid parameters.
    #[test]
    fn prop_biobag_quality_bounded(week in 4u32..=42) {
        let bag = BioBag::new(GestationalWeek::new(week));
        let score = bag.quality_score();
        prop_assert!(score >= 0.0 && score <= 1.0,
            "BioBag quality_score must be in [0, 1], got {} at week {}", score, week);
    }

    // ── Fully Approved Gate Permits High-Benefit Interventions ──────────────
    // A fully approved gate with benefit >= 0.7 should always succeed.
    #[test]
    fn prop_fully_approved_permits_high_benefit(week in 0u32..=42, benefit in 0.7f64..=1.0) {
        let gate = EctogenesisEthicsGate::fully_approved();
        let result = gate.check_intervention(
            GestationalWeek::new(week),
            "high benefit intervention",
            benefit,
        );
        prop_assert!(result.is_ok(),
            "Fully approved gate should permit benefit={} at week {}, got {:?}",
            benefit, week, result);
    }
}
