// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mycelix_bridge_common::consciousness_profile::{
    continuous_vote_weight, evaluate_governance, evaluate_governance_with_reputation,
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier, GovernanceRequirement,
    ReputationState,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Arbitrary input types
// ---------------------------------------------------------------------------

#[derive(Arbitrary, Debug)]
struct FuzzProfile {
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,
}

#[derive(Arbitrary, Debug)]
struct FuzzGovernanceInput {
    profile: FuzzProfile,
    // Credential fields
    issued_at: u64,
    ttl_us: u64,
    // Requirement fields
    min_tier_idx: u8,
    min_identity: Option<f64>,
    min_community: Option<f64>,
    // Current time
    now_us: u64,
}

#[derive(Arbitrary, Debug)]
struct FuzzReputationInput {
    governance: FuzzGovernanceInput,
    rep_score: f64,
    rep_last_updated: u64,
    rep_consecutive_good: u32,
    rep_total_slashes: u32,
    rep_blacklisted: bool,
    rep_blacklisted_since: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tier_from_idx(idx: u8) -> ConsciousnessTier {
    match idx % 5 {
        0 => ConsciousnessTier::Observer,
        1 => ConsciousnessTier::Participant,
        2 => ConsciousnessTier::Citizen,
        3 => ConsciousnessTier::Steward,
        _ => ConsciousnessTier::Guardian,
    }
}

fn make_credential(input: &FuzzGovernanceInput) -> ConsciousnessCredential {
    let profile = ConsciousnessProfile {
        identity: input.profile.identity,
        reputation: input.profile.reputation,
        community: input.profile.community,
        engagement: input.profile.engagement,
    };
    ConsciousnessCredential {
        did: "did:mycelix:fuzz".to_string(),
        profile,
        tier: ConsciousnessTier::Observer, // will be re-derived internally
        issued_at: input.issued_at,
        expires_at: input.issued_at.saturating_add(input.ttl_us),
        issuer: "did:mycelix:fuzz-issuer".to_string(),
        trajectory_commitment: None,
        extensions: HashMap::new(),
    }
}

fn make_requirement(input: &FuzzGovernanceInput) -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: tier_from_idx(input.min_tier_idx),
        min_identity: input.min_identity,
        min_community: input.min_community,
    }
}

// ---------------------------------------------------------------------------
// Fuzz target
// ---------------------------------------------------------------------------

fuzz_target!(|data: FuzzReputationInput| {
    let gov = &data.governance;
    let credential = make_credential(gov);
    let requirement = make_requirement(gov);

    // 1. Exercise evaluate_governance — must never panic
    let result = evaluate_governance(&credential, &requirement, gov.now_us);

    // Invariant: weight_bp must never exceed 10000
    assert!(
        result.weight_bp <= 10000,
        "vote weight {} exceeds 10000 BP",
        result.weight_bp
    );

    // Invariant: if eligible, tier must meet min_tier
    if result.eligible {
        assert!(
            result.tier >= requirement.min_tier,
            "eligible but tier {:?} < required {:?}",
            result.tier,
            requirement.min_tier
        );
    }

    // 2. Exercise evaluate_governance_with_reputation — must never panic
    let rep_state = ReputationState {
        score: data.rep_score,
        last_updated_us: data.rep_last_updated,
        consecutive_good: data.rep_consecutive_good,
        total_slashes: data.rep_total_slashes,
        blacklisted: data.rep_blacklisted,
        blacklisted_since_us: data.rep_blacklisted_since,
    };
    let rep_result =
        evaluate_governance_with_reputation(&credential, &requirement, &rep_state, gov.now_us);

    // Invariant: blacklisted agents must never be eligible
    if rep_state.blacklisted {
        assert!(
            !rep_result.eligible,
            "blacklisted agent should not be eligible"
        );
        assert_eq!(rep_result.weight_bp, 0, "blacklisted weight must be 0");
    }

    // Invariant: slash penalty can only reduce weight
    if rep_state.total_slashes > 0 && !rep_state.blacklisted {
        assert!(
            rep_result.weight_bp <= result.weight_bp,
            "slashed weight {} > unslashed weight {}",
            rep_result.weight_bp,
            result.weight_bp
        );
    }

    // 3. Exercise profile methods — must never panic with arbitrary f64
    let profile = &credential.profile;
    let _score = profile.combined_score();
    let _tier = profile.tier();
    let _clamped = profile.clamped();
    let _valid = profile.is_valid();
    let _weight = profile.vote_weight_continuous();

    // Invariant: clamped profile must have all dimensions in [0, 1]
    let c = profile.clamped();
    assert!((0.0..=1.0).contains(&c.identity));
    assert!((0.0..=1.0).contains(&c.reputation));
    assert!((0.0..=1.0).contains(&c.community));
    assert!((0.0..=1.0).contains(&c.engagement));

    // 4. Exercise continuous_vote_weight with arbitrary parameters
    let sigmoid_result = continuous_vote_weight(
        gov.profile.identity,
        gov.profile.reputation,
        gov.profile.community,
        gov.profile.engagement,
    );
    // Must be finite and non-negative
    assert!(
        sigmoid_result.is_finite() && sigmoid_result >= 0.0,
        "sigmoid returned {} for inputs ({}, {}, {}, {})",
        sigmoid_result,
        gov.profile.identity,
        gov.profile.reputation,
        gov.profile.community,
        gov.profile.engagement,
    );

    // 5. Exercise tier hysteresis — must not panic
    for current_tier_idx in 0..5u8 {
        let current = tier_from_idx(current_tier_idx);
        let _hysteresis_tier = profile.tier_with_hysteresis(current);
    }

    // 6. Exercise tier degradation — must not panic
    for levels in 0..10u32 {
        let _degraded = tier_from_idx(gov.min_tier_idx).degrade(levels);
    }
});
