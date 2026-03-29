// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extended property-based tests for consciousness profile, reputation,
//! offline credentials, saga state machine, and vote weight computation.
//!
//! Complements proptest_gating_invariants.rs (which covers tier/governance).

use proptest::prelude::*;

use mycelix_bridge_common::consciousness_profile::{
    continuous_vote_weight, decay_reputation, ConsciousnessCredential, ConsciousnessProfile,
    ConsciousnessTier, VOTE_WEIGHT_MAX_BP, VOTE_WEIGHT_TEMPERATURE,
};
use mycelix_bridge_common::offline_credential::OfflineCredential;
use mycelix_bridge_common::saga::{self, SagaDefinition, SagaAction, SagaStep, SagaStatus};

const BASE_US: u64 = 1_767_225_600_000_000;
const DAY_US: u64 = 86_400_000_000;
const HOUR_US: u64 = 3_600_000_000;

// ============================================================================
// Strategy helpers
// ============================================================================

fn arb_profile() -> impl Strategy<Value = ConsciousnessProfile> {
    (0.0..=1.0f64, 0.0..=1.0f64, 0.0..=1.0f64, 0.0..=1.0f64).prop_map(
        |(identity, reputation, community, engagement)| ConsciousnessProfile {
            identity,
            reputation,
            community,
            engagement,
        },
    )
}

fn make_credential(profile: ConsciousnessProfile) -> ConsciousnessCredential {
    let tier = profile.clamped().tier();
    ConsciousnessCredential {
        did: "did:mycelix:proptest".into(),
        profile,
        tier,
        issued_at: BASE_US,
        expires_at: BASE_US + DAY_US,
        issuer: "did:mycelix:bridge".into(),
        trajectory_commitment: None,
        extensions: Default::default(),
    }
}

fn tier_ordinal(tier: ConsciousnessTier) -> u8 {
    match tier {
        ConsciousnessTier::Observer => 0,
        ConsciousnessTier::Participant => 1,
        ConsciousnessTier::Citizen => 2,
        ConsciousnessTier::Steward => 3,
        ConsciousnessTier::Guardian => 4,
    }
}

// ============================================================================
// Property 1: Vote weight is monotonically increasing in score
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn vote_weight_monotonic_in_score(
        s1 in 0.0..=1.0f64,
        s2 in 0.0..=1.0f64,
        threshold in 0.1..=0.9f64,
        temperature in 0.01..=0.2f64,
    ) {
        let (lo, hi) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };
        let w_lo = continuous_vote_weight(lo, threshold, temperature, VOTE_WEIGHT_MAX_BP);
        let w_hi = continuous_vote_weight(hi, threshold, temperature, VOTE_WEIGHT_MAX_BP);
        prop_assert!(
            w_lo <= w_hi + 1e-10,
            "Vote weight should be monotonic: w({:.4}) = {:.4} > w({:.4}) = {:.4}",
            lo, w_lo, hi, w_hi
        );
    }
}

// ============================================================================
// Property 2: Vote weight is always in [0, max_weight]
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn vote_weight_bounded(
        score in -1.0..=2.0f64,
        threshold in 0.0..=1.0f64,
        temperature in 0.01..=0.5f64,
        max_weight in 0.0..=100_000.0f64,
    ) {
        let w = continuous_vote_weight(score, threshold, temperature, max_weight);
        prop_assert!(w >= 0.0, "Vote weight should be non-negative: {w}");
        prop_assert!(w <= max_weight + 1e-10, "Vote weight should be <= max: {w} > {max_weight}");
        prop_assert!(w.is_finite(), "Vote weight should be finite: {w}");
    }
}

// ============================================================================
// Property 3: Reputation decay is monotonically decreasing in elapsed days
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn reputation_decay_monotonic(
        profile in arb_profile(),
        days_a in 0.0..=365.0f64,
        days_b in 0.0..=365.0f64,
    ) {
        let (fewer, more) = if days_a <= days_b { (days_a, days_b) } else { (days_b, days_a) };
        let rep_fewer = decay_reputation(&profile, fewer).reputation;
        let rep_more = decay_reputation(&profile, more).reputation;
        prop_assert!(
            rep_more <= rep_fewer + 1e-10,
            "More days should produce lower or equal reputation: days({:.1})={:.4} > days({:.1})={:.4}",
            more, rep_more, fewer, rep_fewer
        );
    }
}

// ============================================================================
// Property 4: Offline credential degradation is monotonic
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn offline_degradation_monotonic(
        score in 0.3..=1.0f64,
        hours_a in 0u64..=200,
        hours_b in 0u64..=200,
    ) {
        let profile = ConsciousnessProfile {
            identity: score, reputation: score, community: score, engagement: score,
        };
        let cred = make_credential(profile);
        let offline = OfflineCredential::new(cred);

        let (fewer_h, more_h) = if hours_a <= hours_b { (hours_a, hours_b) } else { (hours_b, hours_a) };
        let time_fewer = BASE_US + fewer_h * HOUR_US;
        let time_more = BASE_US + more_h * HOUR_US;

        let tier_fewer = offline.effective_tier(time_fewer);
        let tier_more = offline.effective_tier(time_more);

        prop_assert!(
            tier_ordinal(tier_more) <= tier_ordinal(tier_fewer),
            "Longer offline should produce lower or equal tier: {}h={:?} > {}h={:?}",
            more_h, tier_more, fewer_h, tier_fewer
        );
    }
}

// ============================================================================
// Property 5: Saga advance on terminal state is always Complete
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn saga_terminal_advance_idempotent(
        status_idx in 0..3usize,
        now_offset in 0u64..1_000_000,
    ) {
        let terminal_statuses = [
            SagaStatus::Completed,
            SagaStatus::Compensated,
            SagaStatus::CompensationFailed,
        ];
        let status = terminal_statuses[status_idx];

        let step = SagaStep::new("test", "zome", "fn", None, vec![]);
        let mut saga = SagaDefinition::new("prop-saga", vec![step], BASE_US, 0);
        saga.status = status;

        let action = saga::advance(&mut saga, BASE_US + now_offset);
        prop_assert!(
            matches!(action, SagaAction::Complete),
            "Terminal saga should return Complete, got {:?}", action
        );
    }
}

// ============================================================================
// Property 6: NaN in any profile field produces finite combined_score
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn nan_sanitization_combined_score(
        base in arb_profile(),
        nan_field in 0..4usize,
    ) {
        let mut p = base;
        // Inject NaN into one field
        match nan_field {
            0 => p.identity = f64::NAN,
            1 => p.reputation = f64::NAN,
            2 => p.community = f64::NAN,
            3 => p.engagement = f64::NAN,
            _ => unreachable!(),
        }
        let score = p.combined_score();
        prop_assert!(score.is_finite(), "combined_score should be finite with NaN in field {nan_field}: {score}");
        prop_assert!(score >= 0.0 && score <= 1.0, "combined_score should be in [0,1]: {score}");

        // Clamped version should also be safe
        let clamped = p.clamped();
        prop_assert!(clamped.is_valid(), "clamped() should sanitize NaN fields");
        let clamped_score = clamped.combined_score();
        prop_assert!(clamped_score.is_finite() && clamped_score >= 0.0 && clamped_score <= 1.0);
    }
}

// ============================================================================
// Property 7: Hysteresis stability — same score + same tier → same output
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn hysteresis_stability(
        profile in arb_profile(),
        tier_idx in 0..5usize,
    ) {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        let current_tier = tiers[tier_idx];

        // Apply hysteresis twice with same input
        let result1 = profile.tier_with_hysteresis(current_tier);
        let result2 = profile.tier_with_hysteresis(current_tier);

        prop_assert_eq!(
            result1, result2,
            "Hysteresis should be deterministic for same input"
        );

        // Apply hysteresis to the output — should be stable (idempotent)
        let result3 = profile.tier_with_hysteresis(result1);
        prop_assert_eq!(
            result1, result3,
            "Hysteresis should be stable: tier_with_hysteresis(result) == result"
        );
    }
}

// ============================================================================
// Property 8: Infinity inputs produce finite combined_score
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn infinity_sanitization(
        base in arb_profile(),
        inf_field in 0..4usize,
        positive in proptest::bool::ANY,
    ) {
        let mut p = base;
        let inf = if positive { f64::INFINITY } else { f64::NEG_INFINITY };
        match inf_field {
            0 => p.identity = inf,
            1 => p.reputation = inf,
            2 => p.community = inf,
            3 => p.engagement = inf,
            _ => unreachable!(),
        }
        let score = p.combined_score();
        prop_assert!(score.is_finite(), "combined_score should handle infinity in field {inf_field}");
        prop_assert!(score >= 0.0 && score <= 1.0);
    }
}
