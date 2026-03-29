// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for the consciousness profile, credential, tier,
//! governance evaluation, and reputation systems.
//!
//! Complements the security_regression.rs tests which focus on attack vectors.
//! These tests focus on correctness of the core profile/credential/tier logic.

use mycelix_bridge_common::consciousness_profile::*;
use mycelix_bridge_common::consciousness_thresholds::BOOTSTRAP_TTL_US;

// ============================================================================
// Constants
// ============================================================================

const BASE_US: u64 = 1_767_225_600_000_000; // 2026-01-01T00:00:00Z
const HOUR_US: u64 = 3_600_000_000;
const DAY_US: u64 = 86_400_000_000;

fn make_credential(identity: f64, reputation: f64, community: f64, engagement: f64) -> ConsciousnessCredential {
    let profile = ConsciousnessProfile { identity, reputation, community, engagement };
    ConsciousnessCredential {
        did: "did:mycelix:test-agent".into(),
        profile,
        tier: ConsciousnessProfile { identity, reputation, community, engagement }.clamped().tier(),
        issued_at: BASE_US,
        expires_at: BASE_US + DAY_US,
        issuer: "did:mycelix:test-bridge".into(),
        trajectory_commitment: None,
        extensions: Default::default(),
    }
}

// ============================================================================
// ConsciousnessProfile — combined score
// ============================================================================

#[test]
fn test_combined_score_zero_profile() {
    let p = ConsciousnessProfile::zero();
    assert_eq!(p.combined_score(), 0.0);
}

#[test]
fn test_combined_score_max_profile() {
    let p = ConsciousnessProfile {
        identity: 1.0,
        reputation: 1.0,
        community: 1.0,
        engagement: 1.0,
    };
    assert_eq!(p.combined_score(), 1.0);
}

#[test]
fn test_combined_score_weights() {
    // identity=25%, reputation=25%, community=30%, engagement=20%
    let p = ConsciousnessProfile {
        identity: 1.0,
        reputation: 0.0,
        community: 0.0,
        engagement: 0.0,
    };
    assert!((p.combined_score() - 0.25).abs() < 1e-10);

    let p2 = ConsciousnessProfile {
        identity: 0.0,
        reputation: 0.0,
        community: 1.0,
        engagement: 0.0,
    };
    assert!((p2.combined_score() - 0.30).abs() < 1e-10);
}

#[test]
fn test_combined_score_nan_sanitization() {
    let p = ConsciousnessProfile {
        identity: f64::NAN,
        reputation: f64::INFINITY,
        community: f64::NEG_INFINITY,
        engagement: 0.5,
    };
    // NaN/Inf should be sanitized to 0.0, only engagement contributes
    let score = p.combined_score();
    assert!(score.is_finite());
    assert!((score - 0.1).abs() < 1e-10); // 0.2 * 0.5 = 0.1
}

#[test]
fn test_combined_score_clamped() {
    let p = ConsciousnessProfile {
        identity: 2.0,
        reputation: -1.0,
        community: 1.5,
        engagement: -0.5,
    };
    let score = p.combined_score();
    assert!(score >= 0.0 && score <= 1.0);
}

// ============================================================================
// ConsciousnessProfile — tier mapping
// ============================================================================

#[test]
fn test_tier_observer() {
    let p = ConsciousnessProfile { identity: 0.1, reputation: 0.1, community: 0.1, engagement: 0.1 };
    assert_eq!(p.tier(), ConsciousnessTier::Observer);
}

#[test]
fn test_tier_participant() {
    // Combined 0.3 = Participant
    let p = ConsciousnessProfile { identity: 0.3, reputation: 0.3, community: 0.3, engagement: 0.3 };
    assert_eq!(p.tier(), ConsciousnessTier::Participant);
}

#[test]
fn test_tier_citizen() {
    let p = ConsciousnessProfile { identity: 0.4, reputation: 0.4, community: 0.4, engagement: 0.4 };
    assert_eq!(p.tier(), ConsciousnessTier::Citizen);
}

#[test]
fn test_tier_steward() {
    let p = ConsciousnessProfile { identity: 0.6, reputation: 0.6, community: 0.6, engagement: 0.6 };
    assert_eq!(p.tier(), ConsciousnessTier::Steward);
}

#[test]
fn test_tier_guardian() {
    let p = ConsciousnessProfile { identity: 0.8, reputation: 0.8, community: 0.8, engagement: 0.8 };
    assert_eq!(p.tier(), ConsciousnessTier::Guardian);
}

#[test]
fn test_tier_ordering() {
    assert!(ConsciousnessTier::Observer < ConsciousnessTier::Participant);
    assert!(ConsciousnessTier::Participant < ConsciousnessTier::Citizen);
    assert!(ConsciousnessTier::Citizen < ConsciousnessTier::Steward);
    assert!(ConsciousnessTier::Steward < ConsciousnessTier::Guardian);
}

// ============================================================================
// Tier hysteresis
// ============================================================================

#[test]
fn test_tier_hysteresis_prevents_oscillation() {
    // Score at Citizen boundary (0.4): should NOT demote from Citizen to Participant
    let p = ConsciousnessProfile { identity: 0.4, reputation: 0.4, community: 0.4, engagement: 0.4 };
    let tier = p.tier_with_hysteresis(ConsciousnessTier::Citizen);
    assert_eq!(tier, ConsciousnessTier::Citizen, "hysteresis should prevent demotion at boundary");
}

#[test]
fn test_tier_hysteresis_allows_clear_promotion() {
    // Score well above Steward boundary (0.6): should promote from Citizen
    let p = ConsciousnessProfile { identity: 0.7, reputation: 0.7, community: 0.7, engagement: 0.7 };
    let tier = p.tier_with_hysteresis(ConsciousnessTier::Citizen);
    assert_eq!(tier, ConsciousnessTier::Steward);
}

#[test]
fn test_tier_hysteresis_allows_clear_demotion() {
    // Score well below Citizen boundary: should demote from Citizen
    let p = ConsciousnessProfile { identity: 0.2, reputation: 0.2, community: 0.2, engagement: 0.2 };
    let tier = p.tier_with_hysteresis(ConsciousnessTier::Citizen);
    assert!(tier < ConsciousnessTier::Citizen);
}

// ============================================================================
// ConsciousnessProfile — construction methods
// ============================================================================

#[test]
fn test_from_unified_consciousness() {
    let p = ConsciousnessProfile::from_unified_consciousness(0.8, 0.9, 0.7, 0.6);
    assert!((p.engagement - 0.8).abs() < 1e-10);
    assert!((p.identity - 0.9).abs() < 1e-10);
    assert!((p.reputation - 0.7).abs() < 1e-10);
    assert!((p.community - 0.6).abs() < 1e-10);
}

#[test]
fn test_from_unified_consciousness_clamps() {
    let p = ConsciousnessProfile::from_unified_consciousness(1.5, -0.1, 2.0, 0.5);
    assert_eq!(p.engagement, 1.0);
    assert_eq!(p.identity, 0.0);
    assert_eq!(p.reputation, 1.0);
    assert_eq!(p.community, 0.5);
}

#[test]
fn test_from_symthaea_weights() {
    // engagement = 0.35*phi + 0.25*meta + 0.20*coherence + 0.20*care
    let p = ConsciousnessProfile::from_symthaea(
        1.0, 0.0, 0.0, 0.0, // phi only
        0.5, 0.5, 0.5,       // identity/rep/community
    );
    assert!((p.engagement - 0.35).abs() < 1e-10);

    let p2 = ConsciousnessProfile::from_symthaea(
        0.0, 1.0, 0.0, 0.0,
        0.5, 0.5, 0.5,
    );
    assert!((p2.engagement - 0.25).abs() < 1e-10);
}

// ============================================================================
// ConsciousnessProfile — validation
// ============================================================================

#[test]
fn test_is_valid() {
    assert!(ConsciousnessProfile::zero().is_valid());
    assert!(!ConsciousnessProfile { identity: f64::NAN, ..ConsciousnessProfile::zero() }.is_valid());
    assert!(!ConsciousnessProfile { reputation: f64::INFINITY, ..ConsciousnessProfile::zero() }.is_valid());
}

#[test]
fn test_clamped_sanitizes() {
    let p = ConsciousnessProfile {
        identity: f64::NAN,
        reputation: 1.5,
        community: -0.5,
        engagement: f64::INFINITY,
    };
    let c = p.clamped();
    assert!(c.is_valid());
    assert_eq!(c.identity, 0.0);
    assert_eq!(c.reputation, 1.0);
    assert_eq!(c.community, 0.0);
    assert_eq!(c.engagement, 0.0);
}

// ============================================================================
// Continuous vote weight
// ============================================================================

#[test]
fn test_vote_weight_at_threshold() {
    // At threshold: weight = max/2
    let w = continuous_vote_weight(0.4, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!((w - VOTE_WEIGHT_MAX_BP / 2.0).abs() < 1.0);
}

#[test]
fn test_vote_weight_well_above() {
    let w = continuous_vote_weight(0.8, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!(w > 9900.0, "well above threshold should be near max: {w}");
}

#[test]
fn test_vote_weight_well_below() {
    let w = continuous_vote_weight(0.1, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
    assert!(w < 100.0, "well below threshold should be near zero: {w}");
}

#[test]
fn test_vote_weight_nan_safety() {
    assert_eq!(continuous_vote_weight(f64::NAN, 0.4, 0.05, 10000.0), 0.0);
    assert_eq!(continuous_vote_weight(0.5, f64::NAN, 0.05, 10000.0), 0.0);
    assert_eq!(continuous_vote_weight(0.5, 0.4, 0.0, 10000.0), 0.0);
    assert_eq!(continuous_vote_weight(0.5, 0.4, -1.0, 10000.0), 0.0);
}

// ============================================================================
// ConsciousnessCredential
// ============================================================================

#[test]
fn test_credential_expiry() {
    let cred = make_credential(0.5, 0.5, 0.5, 0.5);
    assert!(!cred.is_expired(BASE_US));
    assert!(!cred.is_expired(BASE_US + DAY_US - 1));
    assert!(cred.is_expired(BASE_US + DAY_US));
    assert!(cred.is_expired(BASE_US + DAY_US + 1));
}

#[test]
fn test_credential_from_unified() {
    let cred = ConsciousnessCredential::from_unified_consciousness(
        "did:test:agent".into(),
        0.7, 0.8, 0.6, 0.5,
        "did:test:bridge".into(),
        BASE_US,
    );
    assert_eq!(cred.profile.engagement, 0.7);
    assert_eq!(cred.expires_at, BASE_US + ConsciousnessCredential::DEFAULT_TTL_US);
    assert!(cred.tier >= ConsciousnessTier::Participant);
}

#[test]
fn test_credential_from_symthaea() {
    let cred = ConsciousnessCredential::from_symthaea(
        "did:test:agent".into(),
        0.8, 0.6, 0.7, 0.5, // phi, meta, coherence, care
        0.9, 0.7, 0.6,       // identity, reputation, community
        "did:test:bridge".into(),
        BASE_US,
    );
    // engagement = 0.35*0.8 + 0.25*0.6 + 0.20*0.7 + 0.20*0.5 = 0.28+0.15+0.14+0.10 = 0.67
    assert!((cred.profile.engagement - 0.67).abs() < 1e-10);
}

#[test]
fn test_credential_extensions() {
    let mut cred = make_credential(0.5, 0.5, 0.5, 0.5);
    assert!(cred.get_extension(ExtensionKey::SUBSTRATE_TYPE).is_none());

    cred.set_extension(ExtensionKey::SUBSTRATE_TYPE, vec![1]);
    assert_eq!(cred.get_extension(ExtensionKey::SUBSTRATE_TYPE), Some(&vec![1]));

    let removed = cred.remove_extension(ExtensionKey::SUBSTRATE_TYPE);
    assert_eq!(removed, Some(vec![1]));
    assert!(cred.get_extension(ExtensionKey::SUBSTRATE_TYPE).is_none());
}

#[test]
fn test_credential_trajectory_commitment() {
    let commitment = [42u8; 32];
    let cred = make_credential(0.5, 0.5, 0.5, 0.5).with_trajectory_commitment(commitment);
    assert!(cred.has_trajectory_binding());
    assert_eq!(cred.trajectory_commitment, Some(commitment));
}

// ============================================================================
// Governance evaluation
// ============================================================================

#[test]
fn test_governance_basic_eligible() {
    let cred = make_credential(0.5, 0.5, 0.5, 0.5);
    let req = requirement_for_basic();
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(result.eligible, "Citizen-level agent should pass basic requirement");
    assert!(result.weight_bp > 0);
}

#[test]
fn test_governance_voting_eligible() {
    let cred = make_credential(0.6, 0.6, 0.6, 0.6);
    let req = requirement_for_voting();
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(result.eligible, "Steward should pass voting: {:?}", result.reasons);
}

#[test]
fn test_governance_voting_rejected_low_tier() {
    let cred = make_credential(0.2, 0.2, 0.2, 0.2);
    let req = requirement_for_voting();
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(!result.eligible, "Observer should not pass voting");
}

#[test]
fn test_governance_guardian_requires_high_scores() {
    let cred = make_credential(0.9, 0.9, 0.9, 0.9);
    let req = requirement_for_guardian();
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(result.eligible, "Full Guardian should pass: {:?}", result.reasons);
}

#[test]
fn test_governance_expired_rejected() {
    let cred = make_credential(0.9, 0.9, 0.9, 0.9);
    let req = requirement_for_voting();
    // 2 days after issuance — well past 24h TTL
    let result = evaluate_governance(&cred, &req, BASE_US + 2 * DAY_US);
    assert!(!result.eligible, "Expired credential should be rejected");
}

#[test]
fn test_governance_grace_period_basic_only() {
    let cred = make_credential(0.5, 0.5, 0.5, 0.5);
    // Just past expiry but well within 30-minute grace period
    let in_grace = BASE_US + DAY_US + 60_000_000; // 1 minute past expiry
    assert!(in_grace < cred.expires_at + GRACE_PERIOD_US, "should be within grace period");

    let basic = evaluate_governance(&cred, &requirement_for_basic(), in_grace);
    assert!(basic.eligible, "Grace period should allow basic ops: {:?}", basic.reasons);

    let voting = evaluate_governance(&cred, &requirement_for_voting(), in_grace);
    assert!(!voting.eligible, "Grace period should NOT allow voting");
}

#[test]
fn test_governance_proposal_requirement() {
    let req = requirement_for_proposal();
    let cred = make_credential(0.4, 0.4, 0.4, 0.4);
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(result.eligible, "Citizen should be able to propose: {:?}", result.reasons);
}

#[test]
fn test_governance_constitutional_requirement() {
    let req = requirement_for_constitutional();
    let cred = make_credential(0.7, 0.7, 0.7, 0.7);
    let result = evaluate_governance(&cred, &req, BASE_US + HOUR_US);
    assert!(result.eligible, "Steward should pass constitutional: {:?}", result.reasons);
}

// ============================================================================
// Reputation decay
// ============================================================================

#[test]
fn test_reputation_decay_zero_days() {
    let p = ConsciousnessProfile { identity: 0.5, reputation: 0.8, community: 0.6, engagement: 0.4 };
    let decayed = decay_reputation(&p, 0.0);
    assert!((decayed.reputation - 0.8).abs() < 1e-10);
    // Identity, community, engagement unchanged
    assert_eq!(decayed.identity, 0.5);
    assert_eq!(decayed.community, 0.6);
    assert_eq!(decayed.engagement, 0.4);
}

#[test]
fn test_reputation_decay_30_days() {
    // Decay rate is 0.998/day, so after 30 days: 1.0 * 0.998^30 ≈ 0.9417
    let p = ConsciousnessProfile { identity: 0.5, reputation: 1.0, community: 0.5, engagement: 0.5 };
    let decayed = decay_reputation(&p, 30.0);
    let expected = REPUTATION_DECAY_PER_DAY.powi(30);
    assert!(
        (decayed.reputation - expected).abs() < 0.01,
        "30-day decay should be ~{:.4}, got: {}",
        expected, decayed.reputation
    );
}

#[test]
fn test_reputation_decay_negative_days_no_growth() {
    let p = ConsciousnessProfile { identity: 0.5, reputation: 0.5, community: 0.5, engagement: 0.5 };
    let decayed = decay_reputation(&p, -10.0);
    // Negative days should not increase reputation
    assert!(decayed.reputation <= 0.5 + 1e-10);
}

// ============================================================================
// Governance with reputation
// ============================================================================

#[test]
fn test_governance_with_reputation_blacklisted() {
    let cred = make_credential(0.9, 0.9, 0.9, 0.9);
    let mut rep = ReputationState::new(0.9, BASE_US);
    // Slash reputation below blacklist threshold
    for i in 0..10 {
        rep.slash(BASE_US + i * HOUR_US);
    }
    assert!(rep.score < REPUTATION_BLACKLIST_THRESHOLD);

    let req = requirement_for_basic();
    let result = evaluate_governance_with_reputation(&cred, &req, &rep, BASE_US + HOUR_US);
    assert!(!result.eligible, "Blacklisted agent should be rejected: {:?}", result.reasons);
}

#[test]
fn test_governance_with_reputation_good_standing() {
    let cred = make_credential(0.6, 0.6, 0.6, 0.6);
    let rep = ReputationState::new(0.8, BASE_US);
    let req = requirement_for_voting();
    let result = evaluate_governance_with_reputation(&cred, &req, &rep, BASE_US + HOUR_US);
    assert!(result.eligible, "Good rep + Steward should pass: {:?}", result.reasons);
}

// ============================================================================
// Bootstrap credential
// ============================================================================

#[test]
fn test_bootstrap_credential_basic() {
    let cred = bootstrap_credential(
        "did:mycelix:new-agent".into(),
        0.5, // identity
        BASE_US,
    );
    assert!(cred.tier <= ConsciousnessTier::Citizen);
    assert!(cred.profile.community >= 0.0);
}

#[test]
fn test_bootstrap_eligible() {
    // Small network (< 5 agents) with valid identity (>= 0.25)
    assert!(is_bootstrap_eligible(3, 0.5));
    assert!(!is_bootstrap_eligible(3, 0.0)); // identity too low
    assert!(!is_bootstrap_eligible(10, 0.5)); // too many agents
    assert!(is_bootstrap_eligible(4, 0.25)); // exact threshold
}

#[test]
fn test_bootstrap_governance_evaluation() {
    let cred = bootstrap_credential(
        "did:mycelix:new-agent".into(),
        0.5,
        BASE_US,
    );
    // Bootstrap TTL is 15 minutes — test within that window
    let within_ttl = BASE_US + 600_000_000; // 10 minutes after issuance
    let result = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), within_ttl);
    assert!(result.eligible, "Bootstrap agent should pass basic: {:?}", result.reasons);

    // After 15 minutes, bootstrap credential expires
    let past_ttl = BASE_US + BOOTSTRAP_TTL_US + 1;
    let expired_result = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), past_ttl);
    assert!(!expired_result.eligible, "Expired bootstrap should be rejected");
}

// ============================================================================
// Credential refresh
// ============================================================================

#[test]
fn test_needs_refresh_false_when_fresh() {
    let cred = make_credential(0.5, 0.5, 0.5, 0.5);
    assert!(!needs_refresh(&cred, BASE_US + HOUR_US));
}

#[test]
fn test_needs_refresh_true_near_expiry() {
    let cred = make_credential(0.5, 0.5, 0.5, 0.5);
    // REFRESH_WINDOW_US before expiry
    let near_expiry = cred.expires_at - REFRESH_WINDOW_US / 2;
    assert!(needs_refresh(&cred, near_expiry));
}

// ============================================================================
// Vote weight from profile
// ============================================================================

#[test]
fn test_vote_weight_continuous_gradient() {
    let weights: Vec<f64> = (0..=10)
        .map(|i| {
            let score = i as f64 / 10.0;
            let p = ConsciousnessProfile { identity: score, reputation: score, community: score, engagement: score };
            p.vote_weight_continuous()
        })
        .collect();

    // Should be monotonically increasing
    for i in 1..weights.len() {
        assert!(
            weights[i] >= weights[i - 1],
            "vote weight should be monotonic: w[{}]={} < w[{}]={}",
            i, weights[i], i - 1, weights[i - 1]
        );
    }
}

// ============================================================================
// GovernanceRequirement presets
// ============================================================================

#[test]
fn test_requirement_presets_tier_ordering() {
    assert!(requirement_for_basic().min_tier <= requirement_for_proposal().min_tier);
    assert!(requirement_for_proposal().min_tier <= requirement_for_voting().min_tier);
    assert!(requirement_for_voting().min_tier <= requirement_for_constitutional().min_tier);
    assert!(requirement_for_constitutional().min_tier <= requirement_for_guardian().min_tier);
}
