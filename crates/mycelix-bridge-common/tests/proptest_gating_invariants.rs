// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for the consciousness gating kernel.
//!
//! All functions under test are pure (no HDK dependency).

use proptest::prelude::*;

use mycelix_bridge_common::consciousness_profile::{
    evaluate_bootstrap_governance, evaluate_governance, is_bootstrap_eligible,
    requirement_for_basic, requirement_for_constitutional, requirement_for_guardian,
    requirement_for_proposal, requirement_for_voting, should_audit, ConsciousnessCredential,
    ConsciousnessProfile, ConsciousnessTier, GovernanceRequirement, GRACE_PERIOD_US,
};
use mycelix_bridge_common::consciousness_thresholds::{
    BOOTSTRAP_COMMUNITY_THRESHOLD, BOOTSTRAP_MIN_IDENTITY, BOOTSTRAP_TTL_US,
};

// ============================================================================
// Strategy helpers
// ============================================================================

/// Generate an arbitrary ConsciousnessProfile with dimensions in [0, 1].
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

/// Generate a profile with unclamped (potentially out-of-range) dimensions.
fn arb_unclamped_profile() -> impl Strategy<Value = ConsciousnessProfile> {
    (
        -10.0..=10.0f64,
        -10.0..=10.0f64,
        -10.0..=10.0f64,
        -10.0..=10.0f64,
    )
        .prop_map(
            |(identity, reputation, community, engagement)| ConsciousnessProfile {
                identity,
                reputation,
                community,
                engagement,
            },
        )
}

/// A "reasonable" timestamp base (somewhere around 2025).
const BASE_US: u64 = 1_700_000_000_000_000;

// ============================================================================
// is_bootstrap_eligible (used below, validates the import)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// is_bootstrap_eligible requires agent_count < threshold AND identity >= min.
    #[test]
    fn bootstrap_eligible_boundary(
        agent_count in 0..=(BOOTSTRAP_COMMUNITY_THRESHOLD + 5),
        identity_score in 0.0..=1.0f64,
    ) {
        let result = is_bootstrap_eligible(agent_count, identity_score);
        let expected = agent_count < BOOTSTRAP_COMMUNITY_THRESHOLD
            && identity_score >= BOOTSTRAP_MIN_IDENTITY;
        prop_assert_eq!(result, expected,
            "is_bootstrap_eligible({}, {}) = {}, expected {}",
            agent_count, identity_score, result, expected);
    }
}

/// Build a non-expired credential from a profile.
fn make_credential(profile: ConsciousnessProfile, now_us: u64) -> ConsciousnessCredential {
    let tier = profile.clamped().tier();
    ConsciousnessCredential {
        did: "did:test:prop".to_string(),
        profile,
        tier,
        issued_at: now_us.saturating_sub(60_000_000),
        expires_at: now_us + ConsciousnessCredential::DEFAULT_TTL_US,
        issuer: "test".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

/// Build an expired credential from a profile. Expired `delta_us` before `now_us`.
fn make_expired_credential(
    profile: ConsciousnessProfile,
    now_us: u64,
    delta_us: u64,
) -> ConsciousnessCredential {
    let tier = profile.clamped().tier();
    let expires_at = now_us.saturating_sub(delta_us);
    ConsciousnessCredential {
        did: "did:test:expired".to_string(),
        profile,
        tier,
        issued_at: expires_at.saturating_sub(ConsciousnessCredential::DEFAULT_TTL_US),
        expires_at,
        issuer: "test".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

/// Build a bootstrap credential from an identity score.
fn make_bootstrap_credential(identity_score: f64, now_us: u64) -> ConsciousnessCredential {
    let clamped_identity = identity_score.clamp(0.0, 1.0);
    ConsciousnessCredential {
        did: "did:test:bootstrap".to_string(),
        profile: ConsciousnessProfile {
            identity: clamped_identity,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
        },
        tier: ConsciousnessTier::Participant,
        issued_at: now_us,
        expires_at: now_us.saturating_add(BOOTSTRAP_TTL_US),
        issuer: "did:mycelix:bootstrap".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

fn tier_ordinal(t: ConsciousnessTier) -> u8 {
    match t {
        ConsciousnessTier::Observer => 0,
        ConsciousnessTier::Participant => 1,
        ConsciousnessTier::Citizen => 2,
        ConsciousnessTier::Steward => 3,
        ConsciousnessTier::Guardian => 4,
    }
}

/// All 5 standard requirement presets.
fn all_requirements() -> Vec<GovernanceRequirement> {
    vec![
        requirement_for_basic(),
        requirement_for_proposal(),
        requirement_for_voting(),
        requirement_for_constitutional(),
        requirement_for_guardian(),
    ]
}

// ============================================================================
// Property 1: Tier monotonicity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// If profile A dominates profile B in all dimensions, A's tier >= B's tier.
    #[test]
    fn tier_monotonicity(
        b_identity in 0.0..=1.0f64,
        b_reputation in 0.0..=1.0f64,
        b_community in 0.0..=1.0f64,
        b_engagement in 0.0..=1.0f64,
        delta_identity in 0.0..=1.0f64,
        delta_reputation in 0.0..=1.0f64,
        delta_community in 0.0..=1.0f64,
        delta_engagement in 0.0..=1.0f64,
    ) {
        let b = ConsciousnessProfile {
            identity: b_identity,
            reputation: b_reputation,
            community: b_community,
            engagement: b_engagement,
        };
        // A dominates B: each dimension of A >= corresponding dimension of B.
        let a = ConsciousnessProfile {
            identity: (b_identity + delta_identity).min(1.0),
            reputation: (b_reputation + delta_reputation).min(1.0),
            community: (b_community + delta_community).min(1.0),
            engagement: (b_engagement + delta_engagement).min(1.0),
        };
        prop_assert!(tier_ordinal(a.tier()) >= tier_ordinal(b.tier()),
            "A {:?} (tier {:?}) should >= B {:?} (tier {:?})",
            a, a.tier(), b, b.tier());
    }
}

// ============================================================================
// Property 2: Score-tier consistency
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    /// Combined score maps to the correct tier bracket.
    #[test]
    fn score_tier_consistency(profile in arb_profile()) {
        let score = profile.combined_score();
        let tier = profile.tier();
        let expected = if score >= 0.8 {
            ConsciousnessTier::Guardian
        } else if score >= 0.6 {
            ConsciousnessTier::Steward
        } else if score >= 0.4 {
            ConsciousnessTier::Citizen
        } else if score >= 0.3 {
            ConsciousnessTier::Participant
        } else {
            ConsciousnessTier::Observer
        };
        prop_assert_eq!(tier, expected,
            "score={:.4}, got {:?}, expected {:?}", score, tier, expected);
    }
}

// ============================================================================
// Property 3: Gate soundness
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    /// No credential whose clamped tier < requirement.min_tier can pass
    /// evaluate_governance (on non-expired credentials).
    #[test]
    fn gate_soundness(
        profile in arb_profile(),
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        let credential = make_credential(profile, now_us);
        let clamped_tier = credential.profile.clamped().tier();
        let result = evaluate_governance(&credential, requirement, now_us);
        if tier_ordinal(clamped_tier) < tier_ordinal(requirement.min_tier) {
            prop_assert!(!result.eligible,
                "Tier {:?} < required {:?} but was marked eligible",
                clamped_tier, requirement.min_tier);
        }
    }
}

// ============================================================================
// Property 4: Bootstrap cap
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// evaluate_bootstrap_governance NEVER returns eligible=true for
    /// min_tier > Participant.
    #[test]
    fn bootstrap_cap(
        identity_score in 0.0..=1.0f64,
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        let cred = make_bootstrap_credential(identity_score, now_us);
        let result = evaluate_bootstrap_governance(&cred, requirement, now_us);
        if requirement.min_tier > ConsciousnessTier::Participant {
            prop_assert!(!result.eligible,
                "Bootstrap eligible for {:?} (> Participant)", requirement.min_tier);
        }
    }
}

// ============================================================================
// Property 5: Bootstrap expiry
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Expired bootstrap credentials always return eligible=false.
    #[test]
    fn bootstrap_expiry(
        identity_score in 0.25..=1.0f64,
        extra_us in 1u64..=1_000_000_000u64,
    ) {
        let now_us = BASE_US;
        let cred = make_bootstrap_credential(identity_score, now_us);
        // Evaluate after expiry
        let eval_time = cred.expires_at + extra_us;
        let result = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), eval_time);
        prop_assert!(!result.eligible,
            "Expired bootstrap credential should never be eligible (eval_time={}, expires_at={})",
            eval_time, cred.expires_at);
    }
}

// ============================================================================
// Property 6: Audit completeness (rejections always audited)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    /// Rejections are ALWAYS audited regardless of agent_hash or action_name.
    #[test]
    fn audit_completeness_rejections(
        agent_hash in proptest::collection::vec(any::<u8>(), 0..64),
        action_name in "[a-z_]{1,32}",
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let eligible = false;
        prop_assert!(should_audit(requirement, eligible, &agent_hash, &action_name),
            "Rejection should always be audited");
    }
}

// ============================================================================
// Property 7: Audit high-tier (Citizen/Steward/Guardian always audited)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Citizen, Steward, and Guardian actions are ALWAYS audited when eligible.
    #[test]
    fn audit_high_tier(
        agent_hash in proptest::collection::vec(any::<u8>(), 1..64),
        action_name in "[a-z_]{1,32}",
        tier_idx in 2..5usize,  // Citizen=2, Steward=3, Guardian=4
    ) {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        let requirement = GovernanceRequirement {
            min_tier: tiers[tier_idx],
            min_identity: None,
            min_community: None,
        };
        let eligible = true;
        prop_assert!(should_audit(&requirement, eligible, &agent_hash, &action_name),
            "High-tier action ({:?}) should always be audited", tiers[tier_idx]);
    }
}

// ============================================================================
// Property 8: Audit sampling rate (~10% for basic/proposal approvals)
// ============================================================================

/// Statistical test: over 256 distinct agent hashes, roughly 10% should be
/// audited for basic/proposal-tier approvals with a fixed action name.
#[test]
fn audit_sampling_rate_approximately_10_percent() {
    let action_name = "create_proposal";
    let requirement = requirement_for_basic();
    let eligible = true;

    let mut audited = 0u32;
    for last_byte in 0u8..=255u8 {
        let agent_hash = vec![0u8, 0u8, 0u8, last_byte];
        if should_audit(&requirement, eligible, &agent_hash, action_name) {
            audited += 1;
        }
    }

    // 10% of 256 = ~26. Allow a generous range: 10-50 (4%-20%).
    assert!(
        (10..=50).contains(&audited),
        "Expected ~26 audited out of 256, got {} ({:.1}%)",
        audited,
        audited as f64 / 256.0 * 100.0,
    );
}

/// Additional statistical check: varying both hash and action_name.
#[test]
fn audit_sampling_rate_across_actions() {
    let actions = [
        "create_proposal",
        "view_proposal",
        "comment",
        "list_members",
        "submit_request",
    ];
    let requirement = requirement_for_proposal();
    let eligible = true;

    let total_tests = 256 * actions.len();
    let mut total_audited = 0u32;

    for action in &actions {
        for last_byte in 0u8..=255u8 {
            let agent_hash = vec![last_byte];
            if should_audit(&requirement, eligible, &agent_hash, action) {
                total_audited += 1;
            }
        }
    }

    let pct = total_audited as f64 / total_tests as f64 * 100.0;
    // Should be near 10% (threshold is 26/256). Allow 4%-20%.
    assert!(
        pct > 4.0 && pct < 20.0,
        "Expected ~10% audit rate, got {:.1}% ({}/{})",
        pct,
        total_audited,
        total_tests,
    );
}

// ============================================================================
// Property 9: Weight bounds
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    /// weight_bp is always in [0, 10000].
    #[test]
    fn weight_bounds(
        profile in arb_profile(),
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        let credential = make_credential(profile, now_us);
        let result = evaluate_governance(&credential, requirement, now_us);
        prop_assert!(result.weight_bp <= 10_000,
            "weight_bp {} exceeds 10000", result.weight_bp);
    }

    /// weight_bp bounds for bootstrap governance as well.
    #[test]
    fn weight_bounds_bootstrap(
        identity_score in 0.0..=1.0f64,
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        let cred = make_bootstrap_credential(identity_score, now_us);
        let result = evaluate_bootstrap_governance(&cred, requirement, now_us);
        prop_assert!(result.weight_bp <= 10_000,
            "bootstrap weight_bp {} exceeds 10000", result.weight_bp);
    }
}

// ============================================================================
// Property 10: Clamping idempotence
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    /// profile.clamped().clamped() == profile.clamped()
    #[test]
    fn clamping_idempotence(profile in arb_unclamped_profile()) {
        let once = profile.clamped();
        let twice = once.clamped();
        prop_assert_eq!(once, twice,
            "clamped() is not idempotent for {:?}", profile);
    }
}

// ============================================================================
// Property 11: NaN safety
// ============================================================================

/// Infinity inputs produce valid combined_score after clamping.
/// f64::clamp correctly maps +/-Infinity to the boundary values.
#[test]
fn infinity_safety_combined_score() {
    let cases = [
        (f64::INFINITY, 0.5, 0.5, 0.5),
        (f64::NEG_INFINITY, 0.5, 0.5, 0.5),
        (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY),
        (
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        (f64::INFINITY, f64::NEG_INFINITY, 0.5, 0.5),
    ];

    for (identity, reputation, community, engagement) in cases {
        let profile = ConsciousnessProfile {
            identity,
            reputation,
            community,
            engagement,
        };
        let clamped = profile.clamped();
        let score = clamped.combined_score();
        assert!(
            !score.is_nan(),
            "clamped().combined_score() is NaN for Infinity profile ({}, {}, {}, {})",
            identity,
            reputation,
            community,
            engagement,
        );
        assert!(
            score.is_finite(),
            "clamped().combined_score() is not finite for profile ({}, {}, {}, {})",
            identity,
            reputation,
            community,
            engagement,
        );
        assert!(
            (0.0..=1.0).contains(&score),
            "clamped().combined_score() = {} out of [0,1] for profile ({}, {}, {}, {})",
            score,
            identity,
            reputation,
            community,
            engagement,
        );
    }
}

/// NaN/Infinity inputs are now sanitized by clamped() → 0.0.
/// Previously NaN survived clamp(), but the sanitize() helper now catches it.
/// Callers SHOULD still use is_finite() guards at system boundaries for defense-in-depth.
#[test]
fn nan_sanitized_by_clamped() {
    let nan_cases = [
        (f64::NAN, 0.5, 0.5, 0.5),
        (0.5, f64::NAN, 0.5, 0.5),
        (0.5, 0.5, f64::NAN, 0.5),
        (0.5, 0.5, 0.5, f64::NAN),
    ];
    for (identity, reputation, community, engagement) in nan_cases {
        let profile = ConsciousnessProfile {
            identity,
            reputation,
            community,
            engagement,
        };
        let clamped = profile.clamped();
        let score = clamped.combined_score();
        // NaN dimensions are sanitized to 0.0 by clamped()
        assert!(
            score.is_finite(),
            "Expected clamped() to sanitize NaN but got {} \
             for profile ({}, {}, {}, {})",
            score,
            identity,
            reputation,
            community,
            engagement,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// After clamping, combined_score is always in [0, 1] and finite.
    #[test]
    fn nan_safety_proptest(profile in arb_unclamped_profile()) {
        let clamped = profile.clamped();
        let score = clamped.combined_score();
        prop_assert!(!score.is_nan(), "score is NaN after clamping {:?}", profile);
        prop_assert!(score.is_finite(), "score is infinite after clamping {:?}", profile);
        prop_assert!(score >= 0.0 && score <= 1.0,
            "score {} out of [0,1] after clamping {:?}", score, profile);
    }
}

// ============================================================================
// Property 12: Grace period - Citizen+ never eligible on expired credentials
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Expired credentials with Citizen+ requirements are NEVER eligible,
    /// even during the grace period.
    #[test]
    fn grace_period_citizen_plus_never_eligible(
        profile in arb_profile(),
        // Expired by 1us to just within the grace period
        expired_by_us in 1u64..GRACE_PERIOD_US,
        // Only test Citizen, Steward, Guardian (indices 2, 3, 4)
        req_idx in 2..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        // Create credential that expired `expired_by_us` ago (within grace)
        let credential = make_expired_credential(profile, now_us, expired_by_us);
        assert!(credential.is_expired(now_us), "credential should be expired");
        // Verify we are within the grace window
        assert!(now_us < credential.expires_at.saturating_add(GRACE_PERIOD_US),
            "should be within grace period");

        let result = evaluate_governance(&credential, requirement, now_us);
        prop_assert!(!result.eligible,
            "Expired credential should NOT be eligible for {:?} even in grace period",
            requirement.min_tier);
    }

    /// Expired credentials beyond the grace period are never eligible for anything.
    #[test]
    fn beyond_grace_period_never_eligible(
        profile in arb_profile(),
        extra_us in 0u64..1_000_000_000u64,
        req_idx in 0..5usize,
    ) {
        let reqs = all_requirements();
        let requirement = &reqs[req_idx];
        let now_us = BASE_US;
        // Expired beyond the grace period
        let expired_by = GRACE_PERIOD_US + 1 + extra_us;
        let credential = make_expired_credential(profile, now_us, expired_by);

        let result = evaluate_governance(&credential, requirement, now_us);
        prop_assert!(!result.eligible,
            "Expired (beyond grace) credential should never be eligible for {:?}",
            requirement.min_tier);
    }
}

// ============================================================================
// Property 13: Offline credential — future timestamps degrade
// ============================================================================

use mycelix_bridge_common::offline_credential::{FreshnessAttestation, OfflineCredential};

fn make_offline_credential(
    tier: ConsciousnessTier,
    issued_at: u64,
) -> mycelix_bridge_common::consciousness_profile::ConsciousnessCredential {
    let hours_168 = 168 * 3600 * 1_000_000u64;
    mycelix_bridge_common::consciousness_profile::ConsciousnessCredential {
        did: "did:mycelix:test".to_string(),
        profile: ConsciousnessProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.9,
            engagement: 0.6,
        },
        tier,
        issued_at,
        expires_at: issued_at + hours_168,
        issuer: "did:mycelix:bridge".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Future-dated attestation timestamps beyond clock-skew tolerance cause tier degradation.
    /// Offsets within 15-minute tolerance are accepted as clock skew (not degraded).
    #[test]
    fn future_attestation_degrades(
        // Start beyond the 15-minute clock skew tolerance (900_000_001 microseconds)
        future_offset_us in 900_000_001u64..1_000_000_000_000u64,
    ) {
        let now_us = BASE_US;
        let cred = make_offline_credential(ConsciousnessTier::Guardian, 0);
        let mut offline = OfflineCredential::new(cred);
        // Add a signed attestation in the future (beyond tolerance)
        offline.attestation = Some(FreshnessAttestation {
            attester_did: "did:mycelix:peer".into(),
            timestamp: now_us + future_offset_us,
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![1], // non-empty = "signed"
        });
        let tier = offline.effective_tier(now_us);
        // Should degrade by 2 (Guardian → Citizen)
        prop_assert_eq!(tier, ConsciousnessTier::Citizen,
            "Future attestation beyond tolerance should degrade tier by 2, got {:?}", tier);
    }

    /// Attestation timestamp before credential issuance causes tier degradation (causality).
    #[test]
    fn causality_violation_degrades(
        issued_at in 1_000_000u64..1_000_000_000_000u64,
        backdate in 1u64..1_000_000u64,
    ) {
        let now_us = BASE_US;
        let cred = make_offline_credential(ConsciousnessTier::Guardian, issued_at);
        let mut offline = OfflineCredential::new(cred);
        // Add signed attestation that predates credential issuance
        offline.attestation = Some(FreshnessAttestation {
            attester_did: "did:mycelix:peer".into(),
            timestamp: issued_at.saturating_sub(backdate),
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![1],
        });
        let tier = offline.effective_tier(now_us);
        prop_assert_eq!(tier, ConsciousnessTier::Citizen,
            "Causality-violating attestation should degrade by 2, got {:?}", tier);
    }
}

// ============================================================================
// Property 14: Sub-passport counter overflow safety
// ============================================================================

use mycelix_bridge_common::sub_passport::SubPassport;

#[test]
fn sub_passport_violation_counter_saturates() {
    let mut sp = SubPassport::new(
        "did:mycelix:agent".into(),
        "did:mycelix:human".into(),
        "overflow test".into(),
        1_000_000,
    );
    sp.violation_count = u32::MAX;
    sp.record_violation();
    assert_eq!(
        sp.violation_count,
        u32::MAX,
        "violation_count should saturate at u32::MAX"
    );
}

#[test]
fn sub_passport_correction_counter_saturates() {
    let mut sp = SubPassport::new(
        "did:mycelix:agent".into(),
        "did:mycelix:human".into(),
        "overflow test".into(),
        1_000_000,
    );
    sp.correction_count = u32::MAX;
    sp.record_correction();
    assert_eq!(
        sp.correction_count,
        u32::MAX,
        "correction_count should saturate at u32::MAX"
    );
}

// ============================================================================
// Property 15: Grace period capped at 30 days
// ============================================================================

#[test]
fn grace_period_capped_at_30_days() {
    let cred = make_offline_credential(ConsciousnessTier::Guardian, 0);
    let offline = OfflineCredential::with_grace_hours(cred, 10_000); // way over 720h

    // At 721 hours, should be past 720h grace (capped) and degraded
    let t_721h = 721 * 3600 * 1_000_000;
    let tier = offline.effective_tier(t_721h);
    assert!(
        tier < ConsciousnessTier::Guardian,
        "Grace period should be capped at 720h, but tier was {:?} at 721h",
        tier
    );
}

// ============================================================================
// Property 16: SubPassport signing roundtrip
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// SubPassport survives sign → serialize → deserialize → verify roundtrip.
    #[test]
    fn sub_passport_signing_roundtrip(
        agent_did in "[a-z0-9:]{5,40}",
        delegator_did in "[a-z0-9:]{5,40}",
        key in proptest::collection::vec(any::<u8>(), 32..=32),
    ) {
        let key_arr: [u8; 32] = key.try_into().unwrap();
        let mut sp = SubPassport::new(
            agent_did,
            delegator_did,
            "proptest roundtrip".into(),
            1_000_000,
        );
        sp.sign_blake3(&key_arr);

        // Serialize → Deserialize
        let json = serde_json::to_string(&sp).unwrap();
        let deserialized: SubPassport = serde_json::from_str(&json).unwrap();

        // Verify passes after roundtrip
        prop_assert!(deserialized.verify_blake3(&key_arr),
            "BLAKE3 verification should pass after JSON roundtrip");
    }
}

// ============================================================================
// Property 17: Trajectory commitment binding
// ============================================================================

#[test]
fn trajectory_commitment_binding() {
    let mut cred = ConsciousnessCredential::from_unified_consciousness(
        "did:test".into(),
        0.5,
        0.5,
        0.5,
        0.5,
        "issuer".into(),
        1000,
    );
    assert!(!cred.has_trajectory_binding());
    cred = cred.with_trajectory_commitment([42u8; 32]);
    assert!(cred.has_trajectory_binding());
    assert_eq!(cred.trajectory_commitment, Some([42u8; 32]));
}
