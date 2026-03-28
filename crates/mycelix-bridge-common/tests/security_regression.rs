// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security regression tests for mycelix-bridge-common.
//!
//! Covers identified gaps: replay attacks (expired credentials), clock skew
//! exploitation, credential forgery (attestation tampering), tier boundary
//! enforcement, and overflow/bounds safety.

use std::collections::HashMap;

use mycelix_bridge_common::consciousness_profile::{
    bootstrap_credential, evaluate_bootstrap_governance, evaluate_governance,
    evaluate_governance_with_reputation, requirement_for_basic, requirement_for_guardian,
    requirement_for_voting, ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier,
    ReputationState, GRACE_PERIOD_US, REPUTATION_BLACKLIST_THRESHOLD,
};
use mycelix_bridge_common::consciousness_thresholds::BOOTSTRAP_TTL_US;
use mycelix_bridge_common::offline_credential::{FreshnessAttestation, OfflineCredential};

// ============================================================================
// Helpers
// ============================================================================

/// Base time: 2026-01-01T00:00:00Z in microseconds.
const BASE_US: u64 = 1_767_225_600_000_000;

/// One hour in microseconds.
const HOUR_US: u64 = 3_600_000_000;

/// Clock skew tolerance: 15 minutes in microseconds.
/// Must match CLOCK_SKEW_TOLERANCE_US in offline_credential.rs.
const TOLERANCE_US: u64 = 900 * 1_000_000;

/// 24 hours in microseconds.
const DAY_US: u64 = 86_400_000_000;

fn make_credential(tier: ConsciousnessTier, issued_at: u64) -> ConsciousnessCredential {
    // Profile scores chosen to produce the requested tier via combined_score().
    let (identity, reputation, community, engagement) = match tier {
        ConsciousnessTier::Guardian => (0.9, 0.9, 0.9, 0.9),
        ConsciousnessTier::Steward => (0.7, 0.7, 0.7, 0.7),
        ConsciousnessTier::Citizen => (0.5, 0.5, 0.5, 0.5),
        ConsciousnessTier::Participant => (0.35, 0.35, 0.35, 0.35),
        ConsciousnessTier::Observer => (0.1, 0.1, 0.1, 0.1),
    };
    ConsciousnessCredential {
        did: "did:mycelix:test_agent".to_string(),
        profile: ConsciousnessProfile {
            identity,
            reputation,
            community,
            engagement,
        },
        tier,
        issued_at,
        expires_at: issued_at + DAY_US,
        issuer: "did:mycelix:bridge".to_string(),
        trajectory_commitment: None,
        extensions: HashMap::new(),
    }
}

// ============================================================================
// 1. Replay Prevention Tests
// ============================================================================

/// Credential past its TTL must be rejected for any governance action
/// that requires more than Participant tier. This prevents replay of
/// stale credentials captured from a previous session.
#[test]
fn test_expired_credential_rejected() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    // Evaluate well past expiry (2 days after issuance, TTL is 24h)
    let now = BASE_US + 2 * DAY_US;
    assert!(cred.is_expired(now), "Credential should be expired");

    let result = evaluate_governance(&cred, &requirement_for_voting(), now);
    assert!(
        !result.eligible,
        "Expired credential must be rejected for voting"
    );
    assert_eq!(
        result.tier,
        ConsciousnessTier::Observer,
        "Expired credential outside grace should degrade to Observer"
    );
}

/// Within the 30-minute grace period after expiry, basic (Participant-tier)
/// operations should still work to avoid disruption during credential refresh.
#[test]
fn test_expired_credential_within_grace_period() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    // 1 minute after expiry — within the 30min grace window
    let now = cred.expires_at + 60_000_000;
    assert!(cred.is_expired(now));
    assert!(
        now < cred.expires_at + GRACE_PERIOD_US,
        "Should be within grace period"
    );

    let result = evaluate_governance(&cred, &requirement_for_basic(), now);
    assert!(
        result.eligible,
        "Basic operations should be allowed within grace period"
    );
    // Should have a warning about grace period
    assert!(
        result.reasons.iter().any(|r| r.contains("grace period")),
        "Should warn about grace period usage"
    );
}

/// Voting (Citizen-tier) must be rejected even within the grace period.
/// The grace period is intentionally limited to Participant-tier operations
/// to prevent expired credentials from influencing governance outcomes.
#[test]
fn test_expired_credential_no_grace_for_voting() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    // 1 minute after expiry — within grace window
    let now = cred.expires_at + 60_000_000;
    assert!(cred.is_expired(now));

    let result = evaluate_governance(&cred, &requirement_for_voting(), now);
    assert!(
        !result.eligible,
        "Voting must be rejected even within grace period"
    );
}

// ============================================================================
// 2. Clock Skew Attack Tests
// ============================================================================

/// An offline credential with a future-dated reference time beyond the 1-hour
/// tolerance window should be treated as suspicious and degraded by 2 tiers.
/// This prevents an attacker from setting their clock far ahead to avoid
/// degradation penalties.
#[test]
fn test_future_timestamp_beyond_tolerance_degrades_tier() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    let mut offline = OfflineCredential::new(cred);
    // Set last_online_verification 2 hours in the future relative to now
    let now = BASE_US + DAY_US / 2;
    offline.last_online_verification = now + 2 * HOUR_US;

    let tier = offline.effective_tier(now);
    // Guardian degraded by 2 = Citizen
    assert_eq!(
        tier,
        ConsciousnessTier::Citizen,
        "Future timestamp beyond tolerance should degrade by 2 tiers"
    );
}

/// A reference time within the 15-minute clock skew tolerance should not
/// trigger degradation. This accommodates NTP drift during loadshedding
/// recovery in South Africa.
#[test]
fn test_future_timestamp_within_tolerance_safe() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    let mut offline = OfflineCredential::new(cred);
    // Set last_online_verification 10 minutes ahead — within 15-min tolerance
    let now = BASE_US + DAY_US / 2;
    offline.last_online_verification = now + 10 * 60 * 1_000_000;

    let tier = offline.effective_tier(now);
    assert_eq!(
        tier,
        ConsciousnessTier::Guardian,
        "Within clock skew tolerance should maintain full tier"
    );
}

/// Test the exact boundary of the 15-minute clock skew tolerance.
/// At exactly tolerance, the credential should still be safe (tolerance is
/// checked as > not >=). One microsecond beyond should degrade.
#[test]
fn test_clock_skew_tolerance_boundary() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    let now = BASE_US + DAY_US / 2;

    // Exactly at tolerance boundary (reference_time == now + TOLERANCE)
    let mut offline_at = OfflineCredential::new(cred.clone());
    offline_at.last_online_verification = now + TOLERANCE_US;
    let tier_at = offline_at.effective_tier(now);
    assert_eq!(
        tier_at,
        ConsciousnessTier::Guardian,
        "At exact tolerance boundary should be safe (within tolerance)"
    );

    // One microsecond beyond tolerance
    let mut offline_beyond = OfflineCredential::new(cred);
    offline_beyond.last_online_verification = now + TOLERANCE_US + 1;
    let tier_beyond = offline_beyond.effective_tier(now);
    assert_eq!(
        tier_beyond,
        ConsciousnessTier::Citizen,
        "One microsecond beyond tolerance should trigger 2-tier degradation"
    );
}

// ============================================================================
// 3. Credential Integrity Tests
// ============================================================================

/// If a FreshnessAttestation's BLAKE3 signature does not match the canonical
/// bytes (e.g., data was tampered with after signing), verification must fail.
#[test]
fn test_tampered_attestation_rejected() {
    let key: [u8; 32] = [0xAB; 32];
    let mut attestation = FreshnessAttestation {
        attester_did: "did:mycelix:peer1".to_string(),
        timestamp: BASE_US + HOUR_US,
        tier_at_attestation: ConsciousnessTier::Steward,
        signature: Vec::new(),
    };
    attestation.sign_blake3(&key);
    assert!(
        attestation.verify_blake3(&key),
        "Valid signature should verify"
    );

    // Tamper with the attester DID after signing
    attestation.attester_did = "did:mycelix:attacker".to_string();
    assert!(
        !attestation.verify_blake3(&key),
        "Tampered attestation must fail verification"
    );
}

/// An attestation timestamp that predates the credential's issuance violates
/// causality and should trigger a 2-tier degradation penalty. This prevents
/// backdated attestations from extending credential freshness.
#[test]
fn test_attestation_before_issuance_rejected() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    let mut offline = OfflineCredential::new(cred);

    // Create attestation with timestamp BEFORE credential issuance
    let key: [u8; 32] = [0xCD; 32];
    let mut attestation = FreshnessAttestation {
        attester_did: "did:mycelix:peer1".to_string(),
        timestamp: BASE_US - HOUR_US, // Before issuance!
        tier_at_attestation: ConsciousnessTier::Guardian,
        signature: Vec::new(),
    };
    attestation.sign_blake3(&key);
    offline.attestation = Some(attestation);

    let now = BASE_US + HOUR_US;
    let tier = offline.effective_tier(now);
    // Causality violation → 2-tier degradation (Guardian → Citizen)
    assert_eq!(
        tier,
        ConsciousnessTier::Citizen,
        "Attestation before issuance must trigger causality penalty"
    );
}

// ============================================================================
// 4. Tier Boundary Tests
// ============================================================================

/// Bootstrap credentials are capped at Participant tier. Even with a high
/// identity score that would normally qualify for Guardian, bootstrap should
/// never grant voting or constitutional powers. This prevents a single
/// node in a new community from gaining disproportionate governance power.
#[test]
fn test_guardian_cannot_be_created_via_bootstrap() {
    // High identity score
    let cred = bootstrap_credential("did:mycelix:new_agent".to_string(), 1.0, BASE_US);

    // Verify the tier is capped at Participant
    assert_eq!(
        cred.tier,
        ConsciousnessTier::Participant,
        "Bootstrap credential tier must be capped at Participant"
    );

    // Verify voting is rejected
    let voting_result = evaluate_bootstrap_governance(&cred, &requirement_for_voting(), BASE_US);
    assert!(
        !voting_result.eligible,
        "Bootstrap credential must not grant voting rights"
    );

    // Verify guardian is rejected
    let guardian_result =
        evaluate_bootstrap_governance(&cred, &requirement_for_guardian(), BASE_US);
    assert!(
        !guardian_result.eligible,
        "Bootstrap credential must not grant guardian powers"
    );

    // Verify basic participation IS allowed
    let basic_result = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), BASE_US);
    assert!(
        basic_result.eligible,
        "Bootstrap credential should allow basic participation"
    );
}

/// A blacklisted agent (reputation below REPUTATION_BLACKLIST_THRESHOLD)
/// must be blocked from ALL governance actions, including basic participation.
/// This is the strongest sanction in the graduated enforcement model.
#[test]
fn test_blacklisted_agent_blocked_from_governance() {
    let cred = make_credential(ConsciousnessTier::Guardian, BASE_US);
    let now = BASE_US + HOUR_US;

    // Create a blacklisted reputation state
    let mut rep = ReputationState::new(1.0, BASE_US);
    // Slash repeatedly to drive below blacklist threshold
    for i in 0..10 {
        rep.score = 0.1; // Reset score to keep slashing effective
        rep.slash(BASE_US + i * 1000);
    }
    assert!(
        rep.blacklisted,
        "Agent should be blacklisted after repeated slashing"
    );
    assert!(
        rep.score < REPUTATION_BLACKLIST_THRESHOLD,
        "Score {} should be below blacklist threshold {}",
        rep.score,
        REPUTATION_BLACKLIST_THRESHOLD
    );

    // Basic participation should be blocked
    let basic_result = evaluate_governance_with_reputation(&cred, &requirement_for_basic(), &rep, now);
    assert!(
        !basic_result.eligible,
        "Blacklisted agent must be blocked from basic governance"
    );

    // Voting should also be blocked
    let voting_result =
        evaluate_governance_with_reputation(&cred, &requirement_for_voting(), &rep, now);
    assert!(
        !voting_result.eligible,
        "Blacklisted agent must be blocked from voting"
    );

    // Restoration progress should be visible
    assert!(
        basic_result.restoration_progress < 1.0,
        "Blacklisted agent should show restoration progress < 1.0"
    );
}

// ============================================================================
// 5. Overflow/Bounds Tests
// ============================================================================

/// Slashing an agent whose total_slashes is at u32::MAX must not panic.
/// The saturating_add in slash() should prevent overflow.
#[test]
fn test_max_slashes_saturates() {
    let mut rep = ReputationState::new(0.5, BASE_US);
    rep.total_slashes = u32::MAX;

    // This must not panic
    let score = rep.slash(BASE_US + 1000);
    assert!(score.is_finite(), "Score must remain finite after overflow-safe slash");
    assert_eq!(
        rep.total_slashes,
        u32::MAX,
        "total_slashes should saturate at u32::MAX"
    );
}

/// A ConsciousnessProfile with NaN in the reputation dimension must be
/// safely handled: clamped() should produce 0.0, and governance evaluation
/// should not panic or produce NaN results.
#[test]
fn test_reputation_nan_clamped_to_zero() {
    let profile = ConsciousnessProfile {
        identity: 0.8,
        reputation: f64::NAN,
        community: 0.7,
        engagement: 0.6,
    };

    assert!(!profile.is_valid(), "Profile with NaN should be invalid");

    let clamped = profile.clamped();
    assert_eq!(
        clamped.reputation, 0.0,
        "NaN reputation must be clamped to 0.0"
    );
    assert!(
        clamped.combined_score().is_finite(),
        "Combined score must be finite after clamping"
    );

    // Governance evaluation with NaN profile should not panic
    let cred = ConsciousnessCredential {
        did: "did:mycelix:nan_agent".to_string(),
        profile,
        tier: ConsciousnessTier::Guardian,
        issued_at: BASE_US,
        expires_at: BASE_US + DAY_US,
        issuer: "did:mycelix:bridge".to_string(),
        trajectory_commitment: None,
        extensions: HashMap::new(),
    };

    let result = evaluate_governance(&cred, &requirement_for_basic(), BASE_US + HOUR_US);
    // Should complete without panic — the clamped profile determines eligibility
    assert!(
        result.tier != ConsciousnessTier::Guardian,
        "NaN reputation should reduce effective tier from Guardian"
    );
}

/// ReputationState::new with NaN score must sanitize to 0.0.
#[test]
fn test_reputation_state_nan_initial_score() {
    let rep = ReputationState::new(f64::NAN, BASE_US);
    assert_eq!(rep.score, 0.0, "NaN initial score must be sanitized to 0.0");
    assert!(!rep.blacklisted, "Fresh agent should not be blacklisted");
}

/// ReputationState::new with Infinity score must clamp to 1.0.
#[test]
fn test_reputation_state_infinity_clamped() {
    let rep = ReputationState::new(f64::INFINITY, BASE_US);
    assert_eq!(
        rep.score, 0.0,
        "Infinity initial score must be sanitized (not finite → 0.0)"
    );
}

/// Bootstrap credential TTL expiry should reject even basic operations
/// after the TTL has elapsed, preventing long-lived bootstrap abuse.
#[test]
fn test_bootstrap_credential_ttl_expires() {
    let cred = bootstrap_credential("did:mycelix:new_agent".to_string(), 0.5, BASE_US);

    // Just before expiry — should work
    let before = BASE_US + BOOTSTRAP_TTL_US - 1;
    let result_before = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), before);
    assert!(
        result_before.eligible,
        "Bootstrap should be valid just before TTL"
    );

    // After TTL — should be rejected
    let after = BASE_US + BOOTSTRAP_TTL_US + 1;
    let result_after = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), after);
    assert!(
        !result_after.eligible,
        "Bootstrap credential must be rejected after TTL"
    );
}
