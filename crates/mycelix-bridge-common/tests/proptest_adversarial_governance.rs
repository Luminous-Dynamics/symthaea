// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adversarial property-based tests for the governance security boundary.
//!
//! These tests simulate attacker strategies against the consciousness gating
//! system: credential forgery, clock manipulation, tier escalation,
//! blacklist evasion, and NaN injection attacks.

use proptest::prelude::*;

use mycelix_bridge_common::consciousness_profile::{
    evaluate_governance, evaluate_governance_with_reputation, requirement_for_basic,
    requirement_for_constitutional, requirement_for_guardian, requirement_for_voting,
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier, GovernanceRequirement,
    ReputationState, GRACE_PERIOD_US, REPUTATION_BLACKLIST_THRESHOLD,
};
use mycelix_bridge_common::consciousness_thresholds::BOOTSTRAP_TTL_US;
use mycelix_bridge_common::offline_credential::OfflineCredential;

const BASE_US: u64 = 1_767_225_600_000_000;
const DAY_US: u64 = 86_400_000_000;
const HOUR_US: u64 = 3_600_000_000;

fn make_credential(
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,
    issued_at: u64,
    ttl_us: u64,
) -> ConsciousnessCredential {
    let profile = ConsciousnessProfile {
        identity,
        reputation,
        community,
        engagement,
    };
    ConsciousnessCredential {
        did: "did:mycelix:adversary".into(),
        profile: profile.clone(),
        tier: profile.clamped().tier(),
        issued_at,
        expires_at: issued_at + ttl_us,
        issuer: "did:mycelix:bridge".into(),
        trajectory_commitment: None,
        extensions: Default::default(),
    }
}

fn all_requirements() -> Vec<(&'static str, GovernanceRequirement)> {
    vec![
        ("basic", requirement_for_basic()),
        ("voting", requirement_for_voting()),
        ("constitutional", requirement_for_constitutional()),
        ("guardian", requirement_for_guardian()),
    ]
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
// Attack 1: Expired credential replay
// An attacker replays an old credential after it has expired.
// INVARIANT: Expired credentials NEVER grant access (except grace period for basic).
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn expired_credential_never_grants_voting(
        score in 0.0..=1.0f64,
        hours_past_expiry in 1u64..=8760, // 1h to 1 year past expiry
    ) {
        let cred = make_credential(score, score, score, score, BASE_US, DAY_US);
        let expired_time = cred.expires_at + hours_past_expiry * HOUR_US;

        for (name, req) in all_requirements() {
            if req.min_tier > ConsciousnessTier::Participant {
                let result = evaluate_governance(&cred, &req, expired_time);
                prop_assert!(
                    !result.eligible,
                    "Expired credential should NEVER pass {name} (score={score:.3}, hours_past={hours_past_expiry})"
                );
            }
        }
    }
}

// ============================================================================
// Attack 2: Clock manipulation — future timestamps
// Attacker sets system clock forward to extend credential lifetime.
// INVARIANT: Credential with issued_at in the future degrades or is rejected.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn future_issued_credential_does_not_elevate(
        score in 0.0..=1.0f64,
        future_offset_hours in 1u64..=720, // 1h to 30 days in the future
    ) {
        // Credential issued "in the future" relative to now
        let future_issued = BASE_US + future_offset_hours * HOUR_US;
        let cred = make_credential(score, score, score, score, future_issued, DAY_US);

        // Evaluate at BASE_US (before the credential was "issued")
        let result = evaluate_governance(&cred, &requirement_for_voting(), BASE_US);

        // A future credential evaluated before its issuance is expired (now < issued + ttl is false
        // because expires_at = issued + DAY_US which is in the future, so it's NOT expired)
        // But the tier derived from the score is still the real tier — no elevation
        if result.eligible {
            let expected_tier = ConsciousnessProfile {
                identity: score, reputation: score, community: score, engagement: score,
            }.clamped().tier();
            prop_assert!(
                tier_ordinal(result.tier) <= tier_ordinal(expected_tier),
                "Future credential should not elevate tier above what score warrants"
            );
        }
    }
}

// ============================================================================
// Attack 3: Blacklisted agent attempts governance
// INVARIANT: Blacklisted agents are ALWAYS rejected regardless of score.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn blacklisted_agent_always_rejected(
        score in 0.0..=1.0f64,
        req_idx in 0..4usize,
        slash_count in 5u64..=20,
    ) {
        let cred = make_credential(score, score, score, score, BASE_US, DAY_US);

        // Build a heavily slashed reputation
        let mut rep = ReputationState::new(score, BASE_US);
        for i in 0..slash_count {
            rep.slash(BASE_US + i * HOUR_US);
        }

        let reqs = all_requirements();
        let (name, ref req) = reqs[req_idx];

        let result = evaluate_governance_with_reputation(
            &cred, req, &rep, BASE_US + HOUR_US,
        );

        if rep.blacklisted {
            prop_assert!(
                !result.eligible,
                "Blacklisted agent should NEVER be eligible for {name} (score={score:.3}, slashes={slash_count})"
            );
        }
    }
}

// ============================================================================
// Attack 4: Score inflation via NaN injection
// Attacker crafts a credential with NaN/Infinity in profile fields hoping
// to bypass tier checks.
// INVARIANT: NaN/Infinity fields NEVER produce a higher tier than the valid fields warrant.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn nan_injection_never_elevates_tier(
        base_score in 0.0..=1.0f64,
        nan_field in 0..4usize,
        use_infinity in proptest::bool::ANY,
    ) {
        // Legitimate tier from base score
        let clean_profile = ConsciousnessProfile {
            identity: base_score,
            reputation: base_score,
            community: base_score,
            engagement: base_score,
        };
        let clean_tier = clean_profile.clamped().tier();

        // Inject NaN or Infinity into one field
        let mut poisoned = clean_profile.clone();
        let bad_value = if use_infinity { f64::INFINITY } else { f64::NAN };
        match nan_field {
            0 => poisoned.identity = bad_value,
            1 => poisoned.reputation = bad_value,
            2 => poisoned.community = bad_value,
            3 => poisoned.engagement = bad_value,
            _ => unreachable!(),
        }
        let poisoned_tier = poisoned.clamped().tier();

        // Poisoned tier should NEVER be higher than clean tier
        // (NaN/Inf is sanitized to 0.0, which can only lower the score)
        prop_assert!(
            tier_ordinal(poisoned_tier) <= tier_ordinal(clean_tier),
            "NaN/Inf injection in field {nan_field} elevated tier: {:?} > {:?} (base_score={base_score:.3})",
            poisoned_tier, clean_tier
        );
    }
}

// ============================================================================
// Attack 5: Offline credential extension
// Attacker keeps device offline to avoid credential refresh, hoping the
// degradation model has a loophole at boundary conditions.
// INVARIANT: Offline degradation is monotonically non-increasing in time.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn offline_degradation_no_elevation_at_boundaries(
        score in 0.3..=1.0f64,
    ) {
        let cred = make_credential(score, score, score, score, BASE_US, DAY_US);
        let offline = OfflineCredential::new(cred);

        // Check every hour boundary from 0 to 200 hours
        let mut prev_tier = offline.effective_tier(BASE_US);
        for hour in 1..=200 {
            let current_tier = offline.effective_tier(BASE_US + hour * HOUR_US);
            prop_assert!(
                tier_ordinal(current_tier) <= tier_ordinal(prev_tier),
                "Tier elevated at hour {hour}: {:?} > {:?} (score={score:.3})",
                current_tier, prev_tier
            );
            prev_tier = current_tier;
        }
    }
}

// ============================================================================
// Attack 6: Grace period exploitation
// Attacker tries to use expired credential in grace period for high-privilege ops.
// INVARIANT: Grace period ONLY allows Participant-tier operations, never higher.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn grace_period_never_allows_high_privilege(
        score in 0.0..=1.0f64,
        seconds_past_expiry in 1u64..=1800, // 1s to 30min (grace window)
    ) {
        let cred = make_credential(score, score, score, score, BASE_US, DAY_US);
        let grace_time = cred.expires_at + seconds_past_expiry * 1_000_000;

        // Only check if actually in grace period
        if grace_time < cred.expires_at + GRACE_PERIOD_US {
            // Voting should NEVER be allowed in grace period
            let voting = evaluate_governance(&cred, &requirement_for_voting(), grace_time);
            prop_assert!(
                !voting.eligible,
                "Grace period should NEVER allow voting (score={score:.3}, secs_past={seconds_past_expiry})"
            );

            // Constitutional should NEVER be allowed
            let constitutional = evaluate_governance(&cred, &requirement_for_constitutional(), grace_time);
            prop_assert!(
                !constitutional.eligible,
                "Grace period should NEVER allow constitutional ops"
            );

            // Guardian should NEVER be allowed
            let guardian = evaluate_governance(&cred, &requirement_for_guardian(), grace_time);
            prop_assert!(
                !guardian.eligible,
                "Grace period should NEVER allow guardian ops"
            );
        }
    }
}

// ============================================================================
// Attack 7: Vote weight manipulation
// Attacker crafts extreme profile values hoping to get disproportionate vote weight.
// INVARIANT: Vote weight is bounded by tier's max weight.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn vote_weight_bounded_by_tier(
        score in 0.0..=1.0f64,
    ) {
        let cred = make_credential(score, score, score, score, BASE_US, DAY_US);
        let result = evaluate_governance(&cred, &requirement_for_basic(), BASE_US + HOUR_US);

        if result.eligible {
            // Vote weight should be bounded by the tier's max
            let tier_max_bp = result.tier.vote_weight_bp();
            prop_assert!(
                result.weight_bp <= tier_max_bp,
                "Vote weight {} exceeds tier {:?} max {} (score={score:.3})",
                result.weight_bp, result.tier, tier_max_bp
            );
        }
    }
}
