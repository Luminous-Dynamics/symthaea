// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Recovery zome sweettest integration tests.
//!
//! Tests social recovery setup, initiation, voting, and execution
//! using the Holochain sweettest framework with real conductors.
//!
//! Prerequisites:
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored identity_recovery

extern crate holochain_serialized_bytes;

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Mirror types for deserialization
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct SetupRecoveryInput {
    did: String,
    trustees: Vec<String>,
    threshold: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_lock: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct InitiateRecoveryInput {
    did: String,
    initiator_did: String,
    new_agent: String,
    reason: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct VoteOnRecoveryInput {
    request_id: String,
    trustee_did: String,
    vote: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    comment: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RecoveryConfig {
    did: String,
    trustees: Vec<String>,
    threshold: u32,
    time_lock: u64,
    created_at: i64,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RecoveryRequest {
    request_id: String,
    did: String,
    initiator_did: String,
    new_agent: String,
    reason: String,
    status: String,
    created_at: i64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: Set up social recovery for a DID and verify config.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_setup_recovery_config() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 3).await;
    let alice = &agents[0];
    let bob = &agents[1];
    let carol = &agents[2];

    // Create DIDs for all agents
    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);
    let carol_did = format!("did:mycelix:{}", carol.agent_pubkey);

    // Alice sets up recovery with Bob and Carol as trustees (2-of-2 threshold)
    let setup_input = SetupRecoveryInput {
        did: alice_did.clone(),
        trustees: vec![bob_did.clone(), carol_did.clone()],
        threshold: 2,
        time_lock: Some(3600), // 1 hour time lock
    };
    let _: Record = alice
        .call_zome_fn("recovery", "setup_recovery", setup_input)
        .await;

    // Verify config was stored
    let config: Option<Record> = alice
        .call_zome_fn("recovery", "get_recovery_config", alice_did.clone())
        .await;

    assert!(config.is_some(), "Recovery config should exist after setup");
}

/// Test: Initiate recovery request as trustee.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_initiate_recovery_request() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 3).await;
    let alice = &agents[0];
    let bob = &agents[1];
    let carol = &agents[2];

    // Create DIDs
    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);
    let carol_did = format!("did:mycelix:{}", carol.agent_pubkey);

    // Alice sets up recovery
    let setup_input = SetupRecoveryInput {
        did: alice_did.clone(),
        trustees: vec![bob_did.clone(), carol_did.clone()],
        threshold: 2,
        time_lock: None,
    };
    let _: Record = alice
        .call_zome_fn("recovery", "setup_recovery", setup_input)
        .await;

    wait_for_dht_sync().await;

    // Bob initiates recovery for Alice (Alice lost her key)
    let initiate_input = InitiateRecoveryInput {
        did: alice_did.clone(),
        initiator_did: bob_did.clone(),
        new_agent: "new-agent-pubkey-placeholder".into(),
        reason: "Alice lost access to her device".into(),
    };
    let recovery_record: Record = bob
        .call_zome_fn("recovery", "initiate_recovery", initiate_input)
        .await;

    assert!(
        !recovery_record.action_hashed().hash.as_ref().is_empty(),
        "Recovery request should be created"
    );
}

/// Test: Full recovery voting flow (initiate → vote → check votes).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_recovery_voting_flow() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 3).await;
    let alice = &agents[0];
    let bob = &agents[1];
    let carol = &agents[2];

    // Create DIDs
    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);
    let carol_did = format!("did:mycelix:{}", carol.agent_pubkey);

    // Setup recovery with 2-of-2 threshold
    let setup_input = SetupRecoveryInput {
        did: alice_did.clone(),
        trustees: vec![bob_did.clone(), carol_did.clone()],
        threshold: 2,
        time_lock: None,
    };
    let _: Record = alice
        .call_zome_fn("recovery", "setup_recovery", setup_input)
        .await;

    wait_for_dht_sync().await;

    // Bob initiates recovery
    let initiate_input = InitiateRecoveryInput {
        did: alice_did.clone(),
        initiator_did: bob_did.clone(),
        new_agent: "new-agent-pubkey".into(),
        reason: "Device lost".into(),
    };
    let request_record: Record = bob
        .call_zome_fn("recovery", "initiate_recovery", initiate_input)
        .await;

    let request_id = request_record.action_hashed().hash.to_string();

    wait_for_dht_sync().await;

    // Carol votes to approve
    let vote_input = VoteOnRecoveryInput {
        request_id: request_id.clone(),
        trustee_did: carol_did.clone(),
        vote: true,
        comment: Some("I verified with Alice via phone".into()),
    };
    let _: Record = carol
        .call_zome_fn("recovery", "vote_on_recovery", vote_input)
        .await;

    wait_for_dht_sync().await;

    // Check votes on the request
    let votes: Vec<Record> = bob
        .call_zome_fn("recovery", "get_recovery_votes", request_id.clone())
        .await;

    assert!(
        !votes.is_empty(),
        "Should have at least one vote on the recovery request"
    );
}

/// Test: Owner can cancel a recovery request.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_cancel_recovery_request() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Create DIDs
    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Setup recovery
    let setup_input = SetupRecoveryInput {
        did: alice_did.clone(),
        trustees: vec![bob_did.clone()],
        threshold: 1,
        time_lock: None,
    };
    let _: Record = alice
        .call_zome_fn("recovery", "setup_recovery", setup_input)
        .await;

    wait_for_dht_sync().await;

    // Bob initiates recovery
    let initiate_input = InitiateRecoveryInput {
        did: alice_did.clone(),
        initiator_did: bob_did.clone(),
        new_agent: "new-agent".into(),
        reason: "Thought device was lost".into(),
    };
    let request_record: Record = bob
        .call_zome_fn("recovery", "initiate_recovery", initiate_input)
        .await;

    let request_id = request_record.action_hashed().hash.to_string();

    wait_for_dht_sync().await;

    // Alice cancels (she found her device)
    let _: Record = alice
        .call_zome_fn("recovery", "cancel_recovery", request_id.clone())
        .await;

    // Verify the request is cancelled
    let request: Option<Record> = alice
        .call_zome_fn("recovery", "get_recovery_request", request_id)
        .await;

    assert!(request.is_some(), "Cancelled request should still be retrievable");
}

/// Test: Trustee can check their responsibilities.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_trustee_responsibilities() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Setup recovery with Bob as trustee
    let setup_input = SetupRecoveryInput {
        did: alice_did.clone(),
        trustees: vec![bob_did.clone()],
        threshold: 1,
        time_lock: None,
    };
    let _: Record = alice
        .call_zome_fn("recovery", "setup_recovery", setup_input)
        .await;

    wait_for_dht_sync().await;

    // Bob checks responsibilities
    let responsibilities: Vec<Record> = bob
        .call_zome_fn("recovery", "get_trustee_responsibilities", bob_did.clone())
        .await;

    assert!(
        !responsibilities.is_empty(),
        "Bob should see at least one trustee responsibility (for Alice)"
    );
}
