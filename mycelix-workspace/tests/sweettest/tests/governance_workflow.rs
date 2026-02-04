//! Governance hApp sweettest integration tests.
//!
//! Tests proposal creation, voting, and multi-agent consensus
//! using the Holochain sweettest framework with real conductors.
//!
//! Prerequisites:
//!   cd mycelix-governance && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_governance.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored governance
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

/// Test: Create a proposal and cast a vote.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_proposal_and_vote() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        2,
    )
    .await;

    let proposer = &agents[0];
    let voter = &agents[1];

    // Proposer creates a proposal
    let proposal_input = serde_json::json!({
        "title": "Increase community fund allocation",
        "description": "Proposal to increase the community fund from 5% to 10% of network fees",
        "proposal_type": "Normal",
        "voting_period_hours": 168
    });

    let proposal_record: Record = proposer
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    let proposal_hash = proposal_record.action_hashed().hash.clone();
    assert!(!proposal_hash.as_ref().is_empty(), "Proposal should be created");

    wait_for_dht_sync().await;

    // Voter casts a vote
    let vote_input = serde_json::json!({
        "proposal_id": format!("{:?}", proposal_hash),
        "vote": "Approve",
        "reasoning": "This aligns with community growth goals"
    });

    let vote_record: Record = voter
        .call_zome_fn("voting", "cast_vote", vote_input)
        .await;

    assert!(
        !vote_record.action_hashed().hash.as_ref().is_empty(),
        "Vote should be recorded"
    );
}

/// Test: Multiple voters reach quorum.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_multi_voter_quorum() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        4,
    )
    .await;

    let proposer = &agents[0];

    // Create proposal
    let proposal_input = serde_json::json!({
        "title": "Enable cross-hApp bridge protocol",
        "description": "Activate the bridge protocol for inter-hApp communication",
        "proposal_type": "Fast",
        "voting_period_hours": 24
    });

    let proposal_record: Record = proposer
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    let proposal_hash = proposal_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // All 4 agents vote (3 approve, 1 reject)
    for (i, agent) in agents.iter().enumerate() {
        let vote = if i < 3 { "Approve" } else { "Reject" };
        let vote_input = serde_json::json!({
            "proposal_id": format!("{:?}", proposal_hash),
            "vote": vote,
            "reasoning": format!("Agent {} votes {}", i, vote)
        });

        let _: Record = agent
            .call_zome_fn("voting", "cast_vote", vote_input)
            .await;
    }

    wait_for_dht_sync().await;

    // Query votes for the proposal
    let proposal_id = format!("{:?}", proposal_hash);
    let votes: Vec<Record> = proposer
        .call_zome_fn("voting", "get_proposal_votes", proposal_id)
        .await;

    assert_eq!(votes.len(), 4, "All 4 votes should be recorded");
}

/// Test: Delegation chain — Alice delegates to Bob.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_vote_delegation() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        2,
    )
    .await;

    let alice = &agents[0];
    let bob = &agents[1];

    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice delegates to Bob
    let delegation_input = serde_json::json!({
        "delegate_did": bob_did,
        "scope": "All",
        "duration_hours": 720
    });

    let delegation_record: Record = alice
        .call_zome_fn("voting", "create_delegation", delegation_input)
        .await;

    assert!(
        !delegation_record.action_hashed().hash.as_ref().is_empty(),
        "Delegation should be created"
    );

    wait_for_dht_sync().await;

    // Bob checks effective delegations
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let delegations: Vec<serde_json::Value> = bob
        .call_zome_fn("voting", "get_effective_delegations", alice_did)
        .await;

    assert!(!delegations.is_empty(), "Bob should see Alice's delegation");
}
