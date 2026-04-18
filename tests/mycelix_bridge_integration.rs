#![allow(deprecated)] // Tests use legacy ConsciousnessCredential/Tier for backward-compat bridge testing
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-project integration tests: Symthaea <-> Mycelix consciousness bridge.
//!
//! Tests real type interop between the Symthaea cognitive loop and
//! mycelix-bridge-common consciousness gating. Imports actual types from
//! `mycelix_bridge_common` — no mocks.
//!
//! Forward path:  Symthaea C_unified -> Mycelix engagement dimension
//! Reverse path:  Mycelix 4D profile -> Symthaea social state + neuromod

#![cfg(feature = "mycelix")]

use mycelix_bridge_common::consciousness_profile::{
    continuous_vote_weight, decay_reputation, evaluate_governance_with_reputation,
    VOTE_WEIGHT_MAX_BP, VOTE_WEIGHT_TEMPERATURE,
};
use mycelix_bridge_common::consciousness_profile::{
    evaluate_governance, requirement_for_basic, requirement_for_constitutional,
    requirement_for_guardian, requirement_for_proposal, requirement_for_voting,
    ConsciousnessCredential, ConsciousnessProfile, CivicTier, ReputationState,
    REPUTATION_DECAY_PER_DAY,
};

/// Convenience: a timestamp well before any credential expiry.
const NOW: u64 = 1_000_000_000_000;

/// Build a fresh (non-expired) credential wrapping the given profile.
fn fresh_credential(profile: ConsciousnessProfile) -> ConsciousnessCredential {
    ConsciousnessCredential {
        did: "did:mycelix:test-agent".to_string(),
        profile,
        tier: CivicTier::Observer, // tier is re-derived in evaluate_governance
        issued_at: NOW - 1_000_000,
        expires_at: NOW + 86_400_000_000, // +24h
        issuer: "did:mycelix:test-bridge".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

// ============================================================================
// 1. Governance threshold preset tests
// ============================================================================

#[test]
fn test_governance_thresholds_basic() {
    let req = requirement_for_basic();
    assert_eq!(
        req.min_tier,
        CivicTier::Participant,
        "Basic governance requires Participant tier"
    );
    assert!(
        req.min_identity.is_none(),
        "Basic governance has no identity minimum"
    );
    assert!(
        req.min_community.is_none(),
        "Basic governance has no community minimum"
    );
}

#[test]
fn test_governance_thresholds_proposal() {
    let req = requirement_for_proposal();
    assert_eq!(req.min_tier, CivicTier::Participant);
    assert_eq!(
        req.min_identity,
        Some(0.25),
        "Proposals require identity >= 0.25 (Basic MFA)"
    );
    assert!(req.min_community.is_none());
}

#[test]
fn test_governance_thresholds_voting() {
    let req = requirement_for_voting();
    assert_eq!(
        req.min_tier,
        CivicTier::Citizen,
        "Voting requires Citizen tier"
    );
    assert_eq!(req.min_identity, Some(0.25));
    assert!(req.min_community.is_none());
}

#[test]
fn test_governance_thresholds_constitutional() {
    let req = requirement_for_constitutional();
    assert_eq!(
        req.min_tier,
        CivicTier::Steward,
        "Constitutional changes require Steward tier"
    );
    assert_eq!(
        req.min_identity,
        Some(0.5),
        "Constitutional requires identity >= 0.5"
    );
    assert_eq!(
        req.min_community,
        Some(0.3),
        "Constitutional requires community >= 0.3"
    );
}

#[test]
fn test_governance_thresholds_guardian() {
    let req = requirement_for_guardian();
    assert_eq!(req.min_tier, CivicTier::Guardian);
    assert_eq!(req.min_identity, Some(0.7));
    assert_eq!(req.min_community, Some(0.5));
}

// ============================================================================
// 2. Tier derivation from score
// ============================================================================

#[test]
fn test_consciousness_tier_from_score() {
    // Observer: score < 0.3
    assert_eq!(
        CivicTier::from_score(0.0),
        CivicTier::Observer
    );
    assert_eq!(
        CivicTier::from_score(0.29),
        CivicTier::Observer
    );

    // Participant: 0.3 <= score < 0.4
    assert_eq!(
        CivicTier::from_score(0.3),
        CivicTier::Participant
    );
    assert_eq!(
        CivicTier::from_score(0.39),
        CivicTier::Participant
    );

    // Citizen: 0.4 <= score < 0.6
    assert_eq!(
        CivicTier::from_score(0.4),
        CivicTier::Citizen
    );
    assert_eq!(
        CivicTier::from_score(0.59),
        CivicTier::Citizen
    );

    // Steward: 0.6 <= score < 0.8
    assert_eq!(
        CivicTier::from_score(0.6),
        CivicTier::Steward
    );
    assert_eq!(
        CivicTier::from_score(0.79),
        CivicTier::Steward
    );

    // Guardian: score >= 0.8
    assert_eq!(
        CivicTier::from_score(0.8),
        CivicTier::Guardian
    );
    assert_eq!(
        CivicTier::from_score(1.0),
        CivicTier::Guardian
    );
}

// ============================================================================
// 3. Profile combined_score weighted average
// ============================================================================

#[test]
fn test_profile_combined_score() {
    // Weights: identity=0.25, reputation=0.25, community=0.30, engagement=0.20
    let profile = ConsciousnessProfile {
        identity: 0.8,
        reputation: 0.6,
        community: 0.9,
        engagement: 0.5,
    };
    let expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.30 * 0.9 + 0.20 * 0.5;
    let actual = profile.combined_score();
    assert!(
        (actual - expected).abs() < 1e-10,
        "combined_score should be {expected:.4}, got {actual:.4}"
    );

    // Zero profile
    let zero = ConsciousnessProfile::zero();
    assert_eq!(zero.combined_score(), 0.0);

    // Maximum profile
    let max_profile = ConsciousnessProfile {
        identity: 1.0,
        reputation: 1.0,
        community: 1.0,
        engagement: 1.0,
    };
    assert!((max_profile.combined_score() - 1.0).abs() < 1e-10);
}

// ============================================================================
// 4. evaluate_governance accepts eligible profile
// ============================================================================

#[test]
fn test_evaluate_governance_accepts_eligible() {
    // Profile that exceeds Participant (basic) threshold: combined >= 0.3
    let profile = ConsciousnessProfile {
        identity: 0.5,
        reputation: 0.5,
        community: 0.5,
        engagement: 0.5,
    };
    // combined = 0.5 -> Citizen tier, well above Participant
    let cred = fresh_credential(profile);
    let req = requirement_for_basic();
    let result = evaluate_governance(&cred, &req, NOW);

    assert!(
        result.eligible,
        "Profile with combined=0.5 should be eligible for basic governance, reasons: {:?}",
        result.reasons
    );
    assert!(
        result.weight_bp > 0,
        "Eligible agent should have vote weight > 0"
    );
    assert!(
        result.tier >= CivicTier::Participant,
        "Tier should be at least Participant"
    );
}

// ============================================================================
// 5. evaluate_governance rejects below threshold
// ============================================================================

#[test]
fn test_evaluate_governance_rejects_below_threshold() {
    // Profile with combined < 0.3 -> Observer -> below Participant
    let profile = ConsciousnessProfile {
        identity: 0.1,
        reputation: 0.1,
        community: 0.1,
        engagement: 0.1,
    };
    // combined = 0.1 -> Observer
    let cred = fresh_credential(profile);
    let req = requirement_for_basic();
    let result = evaluate_governance(&cred, &req, NOW);

    assert!(
        !result.eligible,
        "Profile with combined=0.1 should be rejected for basic governance"
    );
    assert_eq!(
        result.weight_bp, 0,
        "Rejected agent should have zero vote weight"
    );
    assert_eq!(result.tier, CivicTier::Observer);
    assert!(
        !result.reasons.is_empty(),
        "Should provide rejection reason"
    );
}

#[test]
fn test_evaluate_governance_rejects_missing_identity() {
    // Profile above Citizen tier but identity below voting requirement (0.25)
    let profile = ConsciousnessProfile {
        identity: 0.1, // below 0.25
        reputation: 0.8,
        community: 0.8,
        engagement: 0.8,
    };
    let cred = fresh_credential(profile);
    let req = requirement_for_voting();
    let result = evaluate_governance(&cred, &req, NOW);

    assert!(
        !result.eligible,
        "Should reject: identity 0.1 below voting requirement 0.25, reasons: {:?}",
        result.reasons
    );
}

// ============================================================================
// 6. Tier vote weight monotonicity
// ============================================================================

#[test]
fn test_tier_vote_weight_monotonic() {
    let tiers = [
        CivicTier::Observer,
        CivicTier::Participant,
        CivicTier::Citizen,
        CivicTier::Steward,
        CivicTier::Guardian,
    ];

    for window in tiers.windows(2) {
        let lower = window[0];
        let higher = window[1];
        assert!(
            higher.vote_weight_bp() >= lower.vote_weight_bp(),
            "Higher tier {:?} ({}bp) must have >= vote weight than {:?} ({}bp)",
            higher,
            higher.vote_weight_bp(),
            lower,
            lower.vote_weight_bp(),
        );
    }

    // Observer specifically has zero weight
    assert_eq!(CivicTier::Observer.vote_weight_bp(), 0);
    // Guardian and Steward both have max weight
    assert_eq!(CivicTier::Guardian.vote_weight_bp(), 10000);
    assert_eq!(CivicTier::Steward.vote_weight_bp(), 10000);
}

// ============================================================================
// 7. Profile clamping and validity
// ============================================================================

#[test]
fn test_profile_roundtrip_clamped() {
    // Extreme values: NaN, Infinity, negative, over 1.0
    let extreme = ConsciousnessProfile {
        identity: f64::NAN,
        reputation: f64::INFINITY,
        community: -0.5,
        engagement: 2.0,
    };

    assert!(!extreme.is_valid(), "Profile with NaN should be invalid");

    let clamped = extreme.clamped();
    assert!(clamped.is_valid(), "Clamped profile should be valid");
    assert_eq!(clamped.identity, 0.0, "NaN should clamp to 0.0");
    assert_eq!(
        clamped.reputation, 0.0,
        "Infinity is not finite, so sanitize() returns 0.0"
    );
    assert_eq!(clamped.community, 0.0, "Negative should clamp to 0.0");
    assert_eq!(clamped.engagement, 1.0, "Over 1.0 should clamp to 1.0");

    // Combined score of clamped profile should be valid
    let score = clamped.combined_score();
    assert!(score.is_finite());
    assert!((0.0..=1.0).contains(&score));
}

// ============================================================================
// 8. Symthaea -> Mycelix bridge: from_unified_consciousness
// ============================================================================

#[test]
fn test_from_unified_consciousness_bridge() {
    // Simulate Symthaea producing C_unified = 0.72
    let c_unified = 0.72;
    let identity = 0.8;
    let reputation = 0.6;
    let community = 0.9;

    let profile = ConsciousnessProfile::from_unified_consciousness(
        c_unified, identity, reputation, community,
    );

    assert_eq!(
        profile.engagement, c_unified,
        "C_unified maps to engagement"
    );
    assert_eq!(profile.identity, identity);
    assert_eq!(profile.reputation, reputation);
    assert_eq!(profile.community, community);

    // Verify tier derivation
    let expected_combined = 0.25 * 0.8 + 0.25 * 0.6 + 0.30 * 0.9 + 0.20 * 0.72;
    let tier = profile.tier();
    let expected_tier = CivicTier::from_score(expected_combined);
    assert_eq!(tier, expected_tier);
}

// ============================================================================
// 9. Reputation decay preserves other dimensions
// ============================================================================

#[test]
fn test_reputation_decay_preserves_dimensions() {
    let profile = ConsciousnessProfile {
        identity: 0.8,
        reputation: 1.0,
        community: 0.7,
        engagement: 0.6,
    };

    let decayed = decay_reputation(&profile, 30.0); // 30 days

    // Reputation should decrease
    assert!(
        decayed.reputation < profile.reputation,
        "Reputation should decay over 30 days: {} -> {}",
        profile.reputation,
        decayed.reputation
    );
    // Other dimensions should be unchanged
    assert_eq!(decayed.identity, profile.identity);
    assert_eq!(decayed.community, profile.community);
    assert_eq!(decayed.engagement, profile.engagement);

    // Decay factor = REPUTATION_DECAY_PER_DAY ^ 30
    let expected_rep = REPUTATION_DECAY_PER_DAY.powf(30.0);
    assert!(
        (decayed.reputation - expected_rep).abs() < 1e-6,
        "Expected reputation {expected_rep:.6}, got {:.6}",
        decayed.reputation
    );
}

// ============================================================================
// 10. Blacklisted reputation blocks governance
// ============================================================================

#[test]
fn test_blacklisted_reputation_blocks_governance() {
    let profile = ConsciousnessProfile {
        identity: 1.0,
        reputation: 1.0,
        community: 1.0,
        engagement: 1.0,
    };
    let cred = fresh_credential(profile);

    let blacklisted = ReputationState {
        score: 0.01, // below REPUTATION_BLACKLIST_THRESHOLD
        last_updated_us: NOW,
        consecutive_good: 0,
        total_slashes: 5,
        blacklisted: true,
        blacklisted_since_us: Some(NOW - 1_000_000),
    };

    let req = requirement_for_basic();
    let result = evaluate_governance_with_reputation(&cred, &req, &blacklisted, NOW);

    assert!(
        !result.eligible,
        "Blacklisted agent should be rejected even with perfect profile"
    );
    assert_eq!(result.weight_bp, 0);
    assert!(
        result.reasons.iter().any(|r| r.contains("Blacklisted")),
        "Should mention blacklist in reasons: {:?}",
        result.reasons
    );
}

// ============================================================================
// 11. Continuous vote weight sigmoid
// ============================================================================

#[test]
fn test_continuous_vote_weight_sigmoid() {
    // At threshold, weight should be ~50% of max
    let at_threshold =
        continuous_vote_weight(0.4, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!(
        (at_threshold - VOTE_WEIGHT_MAX_BP / 2.0).abs() < 1.0,
        "At threshold, weight should be ~{}, got {}",
        VOTE_WEIGHT_MAX_BP / 2.0,
        at_threshold
    );

    // Well above threshold -> approaches max
    let above = continuous_vote_weight(0.8, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!(
        above > 0.99 * VOTE_WEIGHT_MAX_BP,
        "Well above threshold should approach max, got {}",
        above
    );

    // Well below threshold -> approaches 0
    let below = continuous_vote_weight(0.1, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!(
        below < 0.01 * VOTE_WEIGHT_MAX_BP,
        "Well below threshold should approach 0, got {}",
        below
    );

    // Monotonicity
    let scores: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let weights: Vec<f64> = scores
        .iter()
        .map(|s| continuous_vote_weight(*s, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP))
        .collect();
    for w in weights.windows(2) {
        assert!(
            w[1] >= w[0],
            "Vote weight must be monotonically increasing with score"
        );
    }
}

// ============================================================================
// 12. Credential expiry blocks governance
// ============================================================================

#[test]
fn test_expired_credential_rejected() {
    let profile = ConsciousnessProfile {
        identity: 1.0,
        reputation: 1.0,
        community: 1.0,
        engagement: 1.0,
    };
    let mut cred = fresh_credential(profile);
    // Make credential expired well past the 30-minute grace period (GRACE_PERIOD_US = 1_800_000_000)
    cred.expires_at = NOW - 2_000_000_000; // expired 2000s ago, well past 1800s grace window

    let req = requirement_for_basic();
    let result = evaluate_governance(&cred, &req, NOW);

    assert!(!result.eligible, "Expired credential should be rejected");
    assert!(
        result.reasons.iter().any(|r| r.contains("expired")),
        "Should mention expiry in reasons: {:?}",
        result.reasons
    );
}
