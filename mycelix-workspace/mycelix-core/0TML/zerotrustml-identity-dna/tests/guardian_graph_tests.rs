// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the guardian_graph zome
//!
//! These tests use Holochain's sweettest framework to test the guardian graph
//! functionality including circular detection, authorization, and reputation queries.
//!
//! Run with: cargo test --package zerotrustml-identity-dna --test guardian_graph_tests

use hdk::prelude::*;
use holochain::sweettest::*;

/// Test configuration
const DNA_PATH: &str = "zerotrustml_identity.dna";
const ZOME_NAME: &str = "guardian_graph";

/// Helper to create a test DID from an agent pubkey
fn make_did(agent: &AgentPubKey) -> String {
    format!("did:mycelix:{}", hex::encode(&agent.get_raw_39()[0..16]))
}

/// Helper struct for guardian relationship
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GuardianRelationship {
    pub subject_did: String,
    pub guardian_did: String,
    pub relationship_type: String,
    pub weight: f64,
    pub status: String,
    pub metadata: String,
    pub established_at: i64,
    pub expires_at: Option<i64>,
    pub mutual: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddGuardianInput {
    pub relationship: GuardianRelationship,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CycleCheckInput {
    pub subject_did: String,
    pub proposed_guardian_did: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CircularityCheckResult {
    pub has_cycle: bool,
    pub cycle_path: Vec<String>,
    pub cycle_type: String,
    pub depth: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestConsentInput {
    pub subject_did: String,
    pub guardian_did: String,
    pub relationship_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RespondConsentInput {
    pub subject_did: String,
    pub guardian_did: String,
    pub accept: bool,
    pub signature: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GuardianNetworkStrength {
    pub did: String,
    pub total_guardians: u32,
    pub active_guardians: u32,
    pub recovery_capable: bool,
    pub assurance_level: String,
    pub diversity_score: f64,
    pub collective_trust_score: f64,
    pub network_connectivity: f64,
    pub weakness_flags: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AuthorizeRecoveryInput {
    pub subject_did: String,
    pub approving_guardians: Vec<String>,
    pub required_threshold: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecoveryAuthorization {
    pub authorized: bool,
    pub reason: String,
    pub threshold_met: bool,
    pub approval_weight: f64,
    pub required_weight: f64,
}

/// Test: Circular detection - Direct cycle (A guards B, B guards A)
#[tokio::test(flavor = "multi_thread")]
async fn test_direct_circular_detection() {
    // Setup conductors
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let ((alice, _), (bob, _)) = apps.into_tuples();

    let alice_agent = alice.agent_pubkey().clone();
    let bob_agent = bob.agent_pubkey().clone();

    let alice_did = make_did(&alice_agent);
    let bob_did = make_did(&bob_agent);

    // Alice adds Bob as guardian
    let rel1 = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let _: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel1 },
        )
        .await;

    // Now Bob tries to add Alice as guardian - should detect cycle
    let cycle_check: CircularityCheckResult = conductors[1]
        .call(
            &bob.zome(ZOME_NAME),
            "detect_circular_guardianship",
            (bob_did.clone(), alice_did.clone()),
        )
        .await;

    assert!(cycle_check.has_cycle, "Should detect direct cycle");
    assert_eq!(cycle_check.cycle_type, "DIRECT");
}

/// Test: Circular detection - Indirect cycle (A -> B -> C -> A)
#[tokio::test(flavor = "multi_thread")]
async fn test_indirect_circular_detection() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(3).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let cells: Vec<_> = apps.into_iter().collect();

    let alice = &cells[0].0;
    let bob = &cells[1].0;
    let charlie = &cells[2].0;

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());
    let charlie_did = make_did(charlie.agent_pubkey());

    // Create chain: Alice's guardian is Bob
    let rel1 = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let _: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel1 },
        )
        .await;

    // Bob's guardian is Charlie
    let rel2 = GuardianRelationship {
        subject_did: bob_did.clone(),
        guardian_did: charlie_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let _: ActionHash = conductors[1]
        .call(
            &bob.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel2 },
        )
        .await;

    // Now Charlie tries to add Alice as guardian - should detect indirect cycle
    let cycle_check: CircularityCheckResult = conductors[2]
        .call(
            &charlie.zome(ZOME_NAME),
            "detect_circular_guardianship",
            (charlie_did.clone(), alice_did.clone()),
        )
        .await;

    assert!(cycle_check.has_cycle, "Should detect indirect cycle");
    assert_eq!(cycle_check.cycle_type, "INDIRECT");
    assert!(cycle_check.cycle_path.len() >= 3, "Should have path of at least 3 DIDs");
}

/// Test: No circular detection when no cycle exists
#[tokio::test(flavor = "multi_thread")]
async fn test_no_circular_detection() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(3).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let cells: Vec<_> = apps.into_iter().collect();

    let alice = &cells[0].0;
    let bob = &cells[1].0;
    let charlie = &cells[2].0;

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());
    let charlie_did = make_did(charlie.agent_pubkey());

    // Alice's guardian is Bob (no cycle - Charlie is independent)
    let rel1 = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let _: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel1 },
        )
        .await;

    // Check if adding Charlie as guardian for Alice would create a cycle (it shouldn't)
    let cycle_check: CircularityCheckResult = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "detect_circular_guardianship",
            (alice_did.clone(), charlie_did.clone()),
        )
        .await;

    assert!(!cycle_check.has_cycle, "Should not detect cycle");
    assert_eq!(cycle_check.cycle_type, "NONE");
}

/// Test: Self-guardianship is prevented
#[tokio::test(flavor = "multi_thread")]
async fn test_prevent_self_guardianship() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(1).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let alice = apps.into_tuples().0.0;
    let alice_did = make_did(alice.agent_pubkey());

    // Try to add self as guardian
    let rel = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: alice_did.clone(), // Same DID
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let result: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(result.is_err(), "Self-guardianship should be rejected");
}

/// Test: Maximum guardians limit is enforced
#[tokio::test(flavor = "multi_thread")]
async fn test_max_guardians_limit() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    // Need 22 conductors: 1 subject + 21 potential guardians (to exceed limit of 20)
    let mut conductors = SweetConductorBatch::from_standard_config(22).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let cells: Vec<_> = apps.into_iter().collect();

    let subject = &cells[0].0;
    let subject_did = make_did(subject.agent_pubkey());

    // Add 20 guardians (the maximum)
    for i in 1..=20 {
        let guardian = &cells[i].0;
        let guardian_did = make_did(guardian.agent_pubkey());

        let rel = GuardianRelationship {
            subject_did: subject_did.clone(),
            guardian_did: guardian_did.clone(),
            relationship_type: "RECOVERY".to_string(),
            weight: 0.05,
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        let _: ActionHash = conductors[0]
            .call(
                &subject.zome(ZOME_NAME),
                "add_guardian",
                AddGuardianInput { relationship: rel },
            )
            .await;
    }

    // Try to add 21st guardian - should fail
    let extra_guardian = &cells[21].0;
    let extra_guardian_did = make_did(extra_guardian.agent_pubkey());

    let rel = GuardianRelationship {
        subject_did: subject_did.clone(),
        guardian_did: extra_guardian_did,
        relationship_type: "RECOVERY".to_string(),
        weight: 0.05,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let result: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &subject.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(result.is_err(), "Should reject exceeding max guardians");
}

/// Test: Guardian consent workflow
#[tokio::test(flavor = "multi_thread")]
async fn test_guardian_consent_workflow() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let ((alice, _), (bob, _)) = apps.into_tuples();

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());

    // Alice requests Bob as guardian
    let consent_request = RequestConsentInput {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
    };

    let _: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "request_guardian_consent",
            consent_request,
        )
        .await;

    // Bob accepts
    let consent_response = RespondConsentInput {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        accept: true,
        signature: None,
    };

    let _: ActionHash = conductors[1]
        .call(
            &bob.zome(ZOME_NAME),
            "respond_to_consent",
            consent_response,
        )
        .await;

    // Now Alice can add Bob as guardian with ACTIVE status
    let rel = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let result: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(!result.get_raw_39().is_empty(), "Should successfully add guardian after consent");
}

/// Test: Network strength calculation
#[tokio::test(flavor = "multi_thread")]
async fn test_network_strength_calculation() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(6).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let cells: Vec<_> = apps.into_iter().collect();

    let subject = &cells[0].0;
    let subject_did = make_did(subject.agent_pubkey());

    // Add 5 guardians (HIGH assurance level)
    for i in 1..=5 {
        let guardian = &cells[i].0;
        let guardian_did = make_did(guardian.agent_pubkey());

        // Vary relationship types for diversity
        let rel_type = match i % 3 {
            0 => "RECOVERY",
            1 => "ENDORSEMENT",
            _ => "DELEGATION",
        };

        let rel = GuardianRelationship {
            subject_did: subject_did.clone(),
            guardian_did: guardian_did.clone(),
            relationship_type: rel_type.to_string(),
            weight: 0.2,
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        let _: ActionHash = conductors[0]
            .call(
                &subject.zome(ZOME_NAME),
                "add_guardian",
                AddGuardianInput { relationship: rel },
            )
            .await;
    }

    // Get network strength
    let strength: GuardianNetworkStrength = conductors[0]
        .call(
            &subject.zome(ZOME_NAME),
            "get_guardian_network_strength",
            subject_did.clone(),
        )
        .await;

    assert_eq!(strength.active_guardians, 5);
    assert_eq!(strength.assurance_level, "HIGH");
    assert!(strength.recovery_capable);
    assert!(strength.diversity_score > 0.0);
}

/// Test: Recovery authorization with guardian consensus
#[tokio::test(flavor = "multi_thread")]
async fn test_recovery_authorization() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(4).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let cells: Vec<_> = apps.into_iter().collect();

    let subject = &cells[0].0;
    let subject_did = make_did(subject.agent_pubkey());

    // Add 3 guardians with weights 0.3, 0.3, 0.4
    let weights = [0.3, 0.3, 0.4];
    let mut guardian_dids = Vec::new();

    for i in 1..=3 {
        let guardian = &cells[i].0;
        let guardian_did = make_did(guardian.agent_pubkey());
        guardian_dids.push(guardian_did.clone());

        let rel = GuardianRelationship {
            subject_did: subject_did.clone(),
            guardian_did: guardian_did.clone(),
            relationship_type: "RECOVERY".to_string(),
            weight: weights[i - 1],
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        let _: ActionHash = conductors[0]
            .call(
                &subject.zome(ZOME_NAME),
                "add_guardian",
                AddGuardianInput { relationship: rel },
            )
            .await;
    }

    // Test: 2 guardians (weight 0.6) should meet 0.5 threshold
    let auth_input = AuthorizeRecoveryInput {
        subject_did: subject_did.clone(),
        approving_guardians: vec![guardian_dids[0].clone(), guardian_dids[1].clone()],
        required_threshold: 0.5,
    };

    let auth_result: RecoveryAuthorization = conductors[0]
        .call(
            &subject.zome(ZOME_NAME),
            "authorize_recovery",
            auth_input,
        )
        .await;

    assert!(auth_result.authorized);
    assert!(auth_result.threshold_met);
    assert_eq!(auth_result.approval_weight, 0.6);

    // Test: 1 guardian (weight 0.3) should NOT meet 0.5 threshold
    let auth_input_fail = AuthorizeRecoveryInput {
        subject_did: subject_did.clone(),
        approving_guardians: vec![guardian_dids[0].clone()],
        required_threshold: 0.5,
    };

    let auth_result_fail: RecoveryAuthorization = conductors[0]
        .call(
            &subject.zome(ZOME_NAME),
            "authorize_recovery",
            auth_input_fail,
        )
        .await;

    assert!(!auth_result_fail.authorized);
    assert!(!auth_result_fail.threshold_met);
}

/// Test: Invalid weight is rejected
#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_weight_rejected() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let ((alice, _), (bob, _)) = apps.into_tuples();

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());

    // Try with weight > 1.0
    let rel = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 1.5, // Invalid
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let result: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(result.is_err(), "Weight > 1.0 should be rejected");
}

/// Test: Invalid relationship type is rejected
#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_relationship_type_rejected() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let ((alice, _), (bob, _)) = apps.into_tuples();

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());

    let rel = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "INVALID_TYPE".to_string(), // Invalid
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let result: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(result.is_err(), "Invalid relationship type should be rejected");
}

/// Test: Empty guardian graph returns appropriate metrics
#[tokio::test(flavor = "multi_thread")]
async fn test_empty_guardian_graph() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(1).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let alice = apps.into_tuples().0.0;
    let alice_did = make_did(alice.agent_pubkey());

    // Get strength for DID with no guardians
    let strength: GuardianNetworkStrength = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "get_guardian_network_strength",
            alice_did.clone(),
        )
        .await;

    assert_eq!(strength.total_guardians, 0);
    assert_eq!(strength.active_guardians, 0);
    assert_eq!(strength.assurance_level, "NONE");
    assert!(!strength.recovery_capable);
    assert_eq!(strength.diversity_score, 0.0);
    assert!(strength.weakness_flags.contains(&"INSUFFICIENT_GUARDIANS".to_string()));
}

/// Test: Duplicate guardian relationship is rejected
#[tokio::test(flavor = "multi_thread")]
async fn test_duplicate_guardian_rejected() {
    let dna = SweetDnaFile::from_file(DNA_PATH).await.unwrap();
    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors.setup_app("test", &[dna]).await.unwrap();
    let ((alice, _), (bob, _)) = apps.into_tuples();

    let alice_did = make_did(alice.agent_pubkey());
    let bob_did = make_did(bob.agent_pubkey());

    // Add guardian first time
    let rel = GuardianRelationship {
        subject_did: alice_did.clone(),
        guardian_did: bob_did.clone(),
        relationship_type: "RECOVERY".to_string(),
        weight: 0.5,
        status: "ACTIVE".to_string(),
        metadata: "{}".to_string(),
        established_at: 1000000,
        expires_at: None,
        mutual: false,
    };

    let _: ActionHash = conductors[0]
        .call(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel.clone() },
        )
        .await;

    // Try to add same guardian again
    let result: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &alice.zome(ZOME_NAME),
            "add_guardian",
            AddGuardianInput { relationship: rel },
        )
        .await;

    assert!(result.is_err(), "Duplicate guardian should be rejected");
}
