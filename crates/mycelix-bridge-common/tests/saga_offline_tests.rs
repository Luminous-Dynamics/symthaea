// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the saga (distributed transaction) and offline credential modules.

use mycelix_bridge_common::consciousness_profile::{
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier,
};
use mycelix_bridge_common::offline_credential::{FreshnessAttestation, OfflineCredential};
use mycelix_bridge_common::saga::*;

// ============================================================================
// Constants
// ============================================================================

const BASE_US: u64 = 1_767_225_600_000_000;
const HOUR_US: u64 = 3_600_000_000;
const DAY_US: u64 = 86_400_000_000;

fn make_credential(tier: ConsciousnessTier) -> ConsciousnessCredential {
    let score = match tier {
        ConsciousnessTier::Observer => 0.1,
        ConsciousnessTier::Participant => 0.35,
        ConsciousnessTier::Citizen => 0.45,
        ConsciousnessTier::Steward => 0.65,
        ConsciousnessTier::Guardian => 0.85,
    };
    ConsciousnessCredential {
        did: "did:mycelix:test".into(),
        profile: ConsciousnessProfile {
            identity: score,
            reputation: score,
            community: score,
            engagement: score,
        },
        tier,
        issued_at: BASE_US,
        expires_at: BASE_US + DAY_US,
        issuer: "did:mycelix:bridge".into(),
        trajectory_commitment: None,
        extensions: Default::default(),
    }
}

// ============================================================================
// Saga — basic workflow
// ============================================================================

fn make_saga_steps() -> Vec<SagaStep> {
    vec![
        SagaStep::new(
            "finance",
            "payments",
            "create_payment",
            Some("cancel_payment".into()),
            vec![1, 2, 3],
        ),
        SagaStep::new(
            "commons",
            "property_transfer",
            "execute_transfer",
            Some("revert_transfer".into()),
            vec![4, 5, 6],
        ),
        SagaStep::new(
            "identity",
            "did_registry",
            "update_ownership",
            None, // No compensation needed
            vec![7, 8],
        ),
    ]
}

#[test]
fn test_saga_creation() {
    let saga = SagaDefinition::new("property-sale", make_saga_steps(), BASE_US, 5 * 60 * 1_000_000);
    assert_eq!(saga.name, "property-sale");
    assert_eq!(saga.steps.len(), 3);
    assert_eq!(saga.current_step, 0);
    assert_eq!(saga.status, SagaStatus::Created);
    assert_eq!(saga.completed_count(), 0);
    assert!(!saga.is_terminal());
}

#[test]
fn test_saga_happy_path() {
    let mut saga = SagaDefinition::new("test-saga", make_saga_steps(), BASE_US, 0);

    // Step 1: first advance returns Dispatch for step 0
    let action = advance(&mut saga, BASE_US);
    match &action {
        SagaAction::Dispatch { zome, fn_name, .. } => {
            assert_eq!(zome, "payments");
            assert_eq!(fn_name, "create_payment");
        }
        _ => panic!("Expected Dispatch, got {:?}", action),
    }
    assert_eq!(saga.status, SagaStatus::Running);

    // Record success
    record_success(&mut saga, Some(vec![10]));
    assert_eq!(saga.steps[0].status, SagaStepStatus::Completed);

    // Step 2: advance to next step
    let action = advance(&mut saga, BASE_US + 1);
    match &action {
        SagaAction::Dispatch { zome, fn_name, .. } => {
            assert_eq!(zome, "property_transfer");
            assert_eq!(fn_name, "execute_transfer");
        }
        _ => panic!("Expected Dispatch for step 2, got {:?}", action),
    }
    record_success(&mut saga, None);

    // Step 3
    let action = advance(&mut saga, BASE_US + 2);
    match &action {
        SagaAction::Dispatch { zome, fn_name, .. } => {
            assert_eq!(zome, "did_registry");
            assert_eq!(fn_name, "update_ownership");
        }
        _ => panic!("Expected Dispatch for step 3, got {:?}", action),
    }
    record_success(&mut saga, None);

    // Saga should be complete
    let action = advance(&mut saga, BASE_US + 3);
    assert!(matches!(action, SagaAction::Complete));
    assert_eq!(saga.status, SagaStatus::Completed);
    assert_eq!(saga.completed_count(), 3);
    assert!(saga.is_terminal());
}

#[test]
fn test_saga_failure_triggers_compensation() {
    let mut saga = SagaDefinition::new("fail-saga", make_saga_steps(), BASE_US, 0);

    // Complete step 0
    advance(&mut saga, BASE_US);
    record_success(&mut saga, Some(vec![1]));

    // Start step 1 and fail
    advance(&mut saga, BASE_US + 1);
    record_failure(&mut saga, "Transfer denied".into());

    // Advance should trigger compensation
    let action = advance(&mut saga, BASE_US + 2);
    match action {
        SagaAction::Compensate(actions) => {
            // Should compensate step 0 (step 1 has no compensation since it failed)
            assert!(
                !actions.is_empty(),
                "Should have compensation actions for completed steps"
            );
            // The compensation should target the completed step's compensate_fn
            let first = &actions[0];
            assert_eq!(first.fn_name, "cancel_payment");
        }
        _ => panic!("Expected Compensate, got {:?}", action),
    }
}

#[test]
fn test_saga_timeout() {
    let timeout_us = 5 * 60 * 1_000_000; // 5 minutes
    let mut saga = SagaDefinition::new("timeout-saga", make_saga_steps(), BASE_US, timeout_us);

    // Start the saga
    advance(&mut saga, BASE_US);

    // Advance after timeout
    let action = advance(&mut saga, BASE_US + timeout_us + 1);
    assert!(matches!(action, SagaAction::Timeout));
    assert_eq!(saga.status, SagaStatus::Failed);
}

#[test]
fn test_saga_terminal_state_idempotent() {
    let mut saga = SagaDefinition::new("terminal-saga", make_saga_steps(), BASE_US, 0);

    // Complete all steps
    for _ in 0..3 {
        advance(&mut saga, BASE_US);
        record_success(&mut saga, None);
    }
    advance(&mut saga, BASE_US);
    assert_eq!(saga.status, SagaStatus::Completed);

    // Further advances should be no-op
    let action = advance(&mut saga, BASE_US + DAY_US);
    assert!(matches!(action, SagaAction::Complete));
}

#[test]
fn test_saga_mark_compensated() {
    let mut saga = SagaDefinition::new("comp-saga", make_saga_steps(), BASE_US, 0);
    saga.status = SagaStatus::Compensating;
    mark_compensated(&mut saga);
    assert_eq!(saga.status, SagaStatus::Compensated);
    assert!(saga.is_terminal());
}

#[test]
fn test_saga_mark_compensation_failed() {
    let mut saga = SagaDefinition::new("comp-fail", make_saga_steps(), BASE_US, 0);
    saga.status = SagaStatus::Compensating;
    mark_compensation_failed(&mut saga, "DB unreachable".into());
    assert_eq!(saga.status, SagaStatus::CompensationFailed);
    assert!(saga.is_terminal());
}

// ============================================================================
// Saga — pre-built workflows
// ============================================================================

#[test]
fn test_property_sale_saga_structure() {
    let saga = property_sale_saga("prop-hash-123".into(), "did:buyer".into(), 100_000, BASE_US);
    assert!(!saga.steps.is_empty());
    assert_eq!(saga.name, "property-sale");
    // All steps should start Pending
    for step in &saga.steps {
        assert_eq!(step.status, SagaStepStatus::Pending);
    }
}

#[test]
fn test_emergency_response_saga_structure() {
    let saga = emergency_response_saga("incident-001".into(), 33.0, -97.0, BASE_US);
    assert!(!saga.steps.is_empty());
    assert_eq!(saga.name, "emergency-response");
}

#[test]
fn test_course_completion_saga_structure() {
    let saga = course_completion_saga("course-rust-101".into(), "did:student".into(), BASE_US);
    assert!(!saga.steps.is_empty());
    assert_eq!(saga.name, "course-completion");
}

#[test]
fn test_justice_enforcement_saga_structure() {
    let saga = justice_enforcement_saga("case-42".into(), "did:defendant".into(), 5000, BASE_US);
    assert!(!saga.steps.is_empty());
}

#[test]
fn test_carbon_credit_saga_structure() {
    let saga = carbon_credit_saga("route-jnb-cpt".into(), 1400.0, BASE_US);
    assert!(!saga.steps.is_empty());
}

// ============================================================================
// OfflineCredential — tier degradation
// ============================================================================

#[test]
fn test_offline_credential_fresh() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::new(cred);
    // Within 24h: full tier maintained
    let tier = offline.effective_tier(BASE_US + 12 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Guardian);
    assert!(offline.is_usable(BASE_US + 12 * HOUR_US));
}

#[test]
fn test_offline_credential_24h_degrades_one_tier() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::new(cred);
    // 24-72h: drops one tier
    let tier = offline.effective_tier(BASE_US + 36 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Steward);
}

#[test]
fn test_offline_credential_72h_degrades_two_tiers() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::new(cred);
    // 72-168h: drops two tiers
    let tier = offline.effective_tier(BASE_US + 100 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Citizen);
}

#[test]
fn test_offline_credential_7d_falls_to_observer() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::new(cred);
    // >168h: falls to Observer
    let tier = offline.effective_tier(BASE_US + 200 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Observer);
}

#[test]
fn test_offline_credential_degradation_disabled() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let mut offline = OfflineCredential::new(cred);
    offline.degradation_enabled = false;
    // Even after 7+ days, tier should remain (degradation disabled)
    let tier = offline.effective_tier(BASE_US + 200 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Guardian);
}

#[test]
fn test_offline_credential_observer_cannot_degrade_further() {
    let cred = make_credential(ConsciousnessTier::Observer);
    let offline = OfflineCredential::new(cred);
    let tier = offline.effective_tier(BASE_US + 200 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Observer);
}

#[test]
fn test_offline_credential_record_verification_resets() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let mut offline = OfflineCredential::new(cred);

    // Verify at 36h (would normally degrade)
    let verify_time = BASE_US + 36 * HOUR_US;
    offline.record_online_verification(verify_time);

    // Check 12h after verification — should be fresh again
    let tier = offline.effective_tier(verify_time + 12 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Guardian);
}

#[test]
fn test_offline_credential_hours_offline() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::new(cred);
    let hours = offline.hours_offline(BASE_US + 12 * HOUR_US);
    assert!((hours - 12.0).abs() < 0.01);
}

#[test]
fn test_offline_credential_custom_grace_period() {
    let cred = make_credential(ConsciousnessTier::Guardian);
    let offline = OfflineCredential::with_grace_hours(cred, 48);
    // Within custom 48h grace: full tier
    let tier = offline.effective_tier(BASE_US + 36 * HOUR_US);
    assert_eq!(tier, ConsciousnessTier::Guardian, "Custom 48h grace should maintain tier");
}

// ============================================================================
// FreshnessAttestation — BLAKE3 signing
// ============================================================================

#[test]
fn test_freshness_attestation_sign_verify() {
    let key = [42u8; 32];
    let mut attestation = FreshnessAttestation {
        attester_did: "did:mycelix:peer-node".into(),
        timestamp: BASE_US,
        tier_at_attestation: ConsciousnessTier::Steward,
        signature: vec![],
    };

    attestation.sign_blake3(&key);
    assert!(!attestation.signature.is_empty());
    assert!(attestation.verify_blake3(&key));
}

#[test]
fn test_freshness_attestation_wrong_key_fails() {
    let key = [42u8; 32];
    let wrong_key = [99u8; 32];
    let mut attestation = FreshnessAttestation {
        attester_did: "did:mycelix:node".into(),
        timestamp: BASE_US,
        tier_at_attestation: ConsciousnessTier::Citizen,
        signature: vec![],
    };

    attestation.sign_blake3(&key);
    assert!(!attestation.verify_blake3(&wrong_key));
}

#[test]
fn test_freshness_attestation_tampered_data_fails() {
    let key = [42u8; 32];
    let mut attestation = FreshnessAttestation {
        attester_did: "did:mycelix:node".into(),
        timestamp: BASE_US,
        tier_at_attestation: ConsciousnessTier::Steward,
        signature: vec![],
    };

    attestation.sign_blake3(&key);
    // Tamper with the tier
    attestation.tier_at_attestation = ConsciousnessTier::Guardian;
    assert!(!attestation.verify_blake3(&key), "Tampered data should fail verification");
}

#[test]
fn test_freshness_attestation_canonical_bytes_deterministic() {
    let a1 = FreshnessAttestation {
        attester_did: "did:mycelix:node".into(),
        timestamp: 12345,
        tier_at_attestation: ConsciousnessTier::Citizen,
        signature: vec![],
    };
    let a2 = a1.clone();
    assert_eq!(a1.canonical_bytes(), a2.canonical_bytes());
}

#[test]
fn test_freshness_attestation_different_tiers_different_bytes() {
    let a1 = FreshnessAttestation {
        attester_did: "did:mycelix:node".into(),
        timestamp: 12345,
        tier_at_attestation: ConsciousnessTier::Citizen,
        signature: vec![],
    };
    let a2 = FreshnessAttestation {
        tier_at_attestation: ConsciousnessTier::Steward,
        ..a1.clone()
    };
    assert_ne!(a1.canonical_bytes(), a2.canonical_bytes());
}
