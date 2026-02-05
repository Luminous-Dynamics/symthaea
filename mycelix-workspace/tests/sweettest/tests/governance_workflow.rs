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

    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Proposer creates a proposal (must match Proposal struct exactly)
    let proposal_input = serde_json::json!({
        "id": "MIP-001",
        "title": "Increase community fund allocation",
        "description": "Proposal to increase the community fund from 5% to 10% of network fees",
        "proposal_type": "Funding",
        "author": format!("did:mycelix:{}", proposer.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let proposal_record: Record = proposer
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    let proposal_hash = proposal_record.action_hashed().hash.clone();
    assert!(!proposal_hash.as_ref().is_empty(), "Proposal should be created");

    wait_for_dht_sync().await;

    // Voter casts a vote (must match CastVoteInput struct)
    let vote_input = serde_json::json!({
        "proposal_id": "MIP-001",
        "voter_did": format!("did:mycelix:{}", voter.agent_pubkey),
        "choice": "For",
        "reason": "This aligns with community growth goals"
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

    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 24 * 60 * 60 * 1_000_000);

    // Create proposal
    let proposal_input = serde_json::json!({
        "id": "MIP-002",
        "title": "Enable cross-hApp bridge protocol",
        "description": "Activate the bridge protocol for inter-hApp communication",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", proposer.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = proposer
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    // All 4 agents vote (3 For, 1 Against)
    for (i, agent) in agents.iter().enumerate() {
        let choice = if i < 3 { "For" } else { "Against" };
        let vote_input = serde_json::json!({
            "proposal_id": "MIP-002",
            "voter_did": format!("did:mycelix:{}", agent.agent_pubkey),
            "choice": choice,
            "reason": format!("Agent {} votes {}", i, choice)
        });

        let _: Record = agent
            .call_zome_fn("voting", "cast_vote", vote_input)
            .await;
    }

    // Extra sync time for 4-conductor gossip
    wait_for_dht_sync().await;
    wait_for_dht_sync().await;

    // Query votes for the proposal
    let votes: Vec<Record> = proposer
        .call_zome_fn("voting", "get_proposal_votes", "MIP-002".to_string())
        .await;

    assert!(votes.len() >= 3, "At least 3 of 4 votes should be propagated (got {})", votes.len());
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

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice delegates to Bob (must match CreateDelegationInput struct)
    let delegation_input = serde_json::json!({
        "delegator_did": alice_did,
        "delegate_did": bob_did,
        "percentage": 1.0,
        "topics": null,
        "tier_filter": null,
        "decay": null,
        "transitive": null,
        "max_chain_depth": null,
        "expires": null
    });

    let delegation_record: Record = alice
        .call_zome_fn("voting", "create_delegation", delegation_input)
        .await;

    assert!(
        !delegation_record.action_hashed().hash.as_ref().is_empty(),
        "Delegation should be created"
    );

    wait_for_dht_sync().await;

    // Bob checks effective delegations for Alice
    let delegations: Vec<serde_json::Value> = bob
        .call_zome_fn("voting", "get_effective_delegations", alice_did.clone())
        .await;

    assert!(!delegations.is_empty(), "Bob should see Alice's delegation");
}
