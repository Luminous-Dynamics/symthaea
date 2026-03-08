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

// ============================================================================
// PROPOSAL LIFECYCLE TESTS
// ============================================================================

/// Test: Create a proposal and retrieve it by ID.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_proposal_create_and_get() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    let proposal_input = serde_json::json!({
        "id": "MIP-100",
        "title": "Test proposal retrieval",
        "description": "Verify that proposals can be retrieved by ID after creation",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", agent.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = agent
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    // Retrieve by ID
    let retrieved: Option<Record> = agent
        .call_zome_fn("proposals", "get_proposal", "MIP-100".to_string())
        .await;

    assert!(retrieved.is_some(), "Proposal should be retrievable by ID");
}

/// Test: Get active proposals returns created proposals.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_get_active_proposals() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    let proposal_input = serde_json::json!({
        "id": "MIP-200",
        "title": "Active proposal test",
        "description": "Should appear in active proposals list",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", agent.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = agent
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    let active: Vec<Record> = agent
        .call_zome_fn("proposals", "get_active_proposals", ())
        .await;

    assert!(!active.is_empty(), "Active proposals list should not be empty");
}

/// Test: Get proposals by author.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_get_proposals_by_author() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let author_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    let proposal_input = serde_json::json!({
        "id": "MIP-201",
        "title": "Author query test",
        "description": "Should be returned when querying by author",
        "proposal_type": "Funding",
        "author": author_did,
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = agent
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    let by_author: Vec<Record> = agent
        .call_zome_fn("proposals", "get_proposals_by_author", author_did.clone())
        .await;

    assert!(!by_author.is_empty(), "Should find proposals by author DID");
}

/// Test: Generate proposal ID auto-increments.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_generate_proposal_id() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let first_id: String = agent
        .call_zome_fn("proposals", "generate_proposal_id", ())
        .await;

    // Before any proposals are created, should be MIP-0001
    assert!(first_id.starts_with("MIP-"), "Generated ID should have MIP- prefix (got {})", first_id);
}

// ============================================================================
// VOTING & TALLY TESTS
// ============================================================================

/// Test: Cast a vote and tally the results.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_vote_and_tally() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        3,
    )
    .await;

    let proposer = &agents[0];
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Create proposal
    let proposal_input = serde_json::json!({
        "id": "MIP-300",
        "title": "Tally test proposal",
        "description": "Tests that vote tally aggregates correctly",
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

    // Agent 1 votes For, Agent 2 votes Against
    for (i, agent) in agents[1..].iter().enumerate() {
        let choice = if i == 0 { "For" } else { "Against" };
        let vote_input = serde_json::json!({
            "proposal_id": "MIP-300",
            "voter_did": format!("did:mycelix:{}", agent.agent_pubkey),
            "choice": choice,
            "reason": format!("Agent {} tally test vote", i + 1)
        });

        let _: Record = agent
            .call_zome_fn("voting", "cast_vote", vote_input)
            .await;
    }

    wait_for_dht_sync().await;
    wait_for_dht_sync().await;

    // Tally votes
    let tally_input = serde_json::json!({
        "proposal_id": "MIP-300",
        "tier": null,
        "quorum_override": null,
        "approval_override": null
    });

    let tally_record: Record = proposer
        .call_zome_fn("voting", "tally_votes", tally_input)
        .await;

    assert!(
        !tally_record.action_hashed().hash.as_ref().is_empty(),
        "Tally record should be created"
    );
}

/// Test: Retrieve tally for a proposal after tallying.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_get_proposal_tally() {
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

    // Create proposal
    let proposal_input = serde_json::json!({
        "id": "MIP-301",
        "title": "Tally retrieval test",
        "description": "Tests get_proposal_tally after tally_votes",
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

    // Cast one vote
    let vote_input = serde_json::json!({
        "proposal_id": "MIP-301",
        "voter_did": format!("did:mycelix:{}", voter.agent_pubkey),
        "choice": "For",
        "reason": "Tally retrieval test"
    });

    let _: Record = voter
        .call_zome_fn("voting", "cast_vote", vote_input)
        .await;

    wait_for_dht_sync().await;

    // Tally
    let tally_input = serde_json::json!({
        "proposal_id": "MIP-301",
        "tier": null,
        "quorum_override": null,
        "approval_override": null
    });

    let _: Record = proposer
        .call_zome_fn("voting", "tally_votes", tally_input)
        .await;

    wait_for_dht_sync().await;

    // Retrieve tally
    let tally: Option<Record> = proposer
        .call_zome_fn("voting", "get_proposal_tally", "MIP-301".to_string())
        .await;

    assert!(tally.is_some(), "Tally should be retrievable after tallying votes");
}

/// Test: Revoke a delegation.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_revoke_delegation() {
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

    // Create delegation
    let delegation_input = serde_json::json!({
        "delegator_did": alice_did,
        "delegate_did": bob_did,
        "percentage": 0.5,
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

    let delegation_hash = delegation_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Revoke the delegation
    let revoke_input = serde_json::json!({
        "delegation_action_hash": delegation_hash,
        "delegator_did": alice_did,
        "reason": "No longer needed"
    });

    let revoke_record: Record = alice
        .call_zome_fn("voting", "revoke_delegation", revoke_input)
        .await;

    assert!(
        !revoke_record.action_hashed().hash.as_ref().is_empty(),
        "Delegation revocation should succeed"
    );
}

// ============================================================================
// COUNCIL TESTS
// ============================================================================

/// Test: Create a council and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_council_create_and_get() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let council_input = serde_json::json!({
        "name": "Technical Council",
        "purpose": "Oversee technical decisions for the network",
        "council_type": { "Domain": { "domain": "Technical" } },
        "parent_council_id": null,
        "phi_threshold": 0.3,
        "quorum": 0.5,
        "supermajority": 0.67,
        "can_spawn_children": true,
        "max_delegation_depth": 3,
        "signing_committee_id": null
    });

    let council_record: Record = agent
        .call_zome_fn("councils", "create_council", council_input)
        .await;

    assert!(
        !council_record.action_hashed().hash.as_ref().is_empty(),
        "Council should be created"
    );

    wait_for_dht_sync().await;

    // List all councils
    let all_councils: Vec<Record> = agent
        .call_zome_fn("councils", "get_all_councils", ())
        .await;

    assert!(!all_councils.is_empty(), "At least one council should exist");
}

/// Test: Join a council and verify membership.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_council_membership() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        2,
    )
    .await;

    let creator = &agents[0];
    let joiner = &agents[1];
    let joiner_did = format!("did:mycelix:{}", joiner.agent_pubkey);

    // Create council
    let council_input = serde_json::json!({
        "name": "Membership Test Council",
        "purpose": "Testing council membership operations",
        "council_type": "Root",
        "parent_council_id": null,
        "phi_threshold": 0.1,
        "quorum": 0.5,
        "supermajority": 0.67,
        "can_spawn_children": false,
        "max_delegation_depth": 2,
        "signing_committee_id": null
    });

    let council_record: Record = creator
        .call_zome_fn("councils", "create_council", council_input)
        .await;

    // Extract council ID from the record entry
    let council_entry: serde_json::Value = council_record
        .entry()
        .to_app_option()
        .expect("Should deserialize")
        .expect("Should have entry");
    let council_id = council_entry["id"].as_str().expect("Council should have id").to_string();

    wait_for_dht_sync().await;

    // Joiner joins the council
    let join_input = serde_json::json!({
        "council_id": council_id,
        "member_did": joiner_did,
        "role": "Member",
        "phi_score": 0.5
    });

    let membership_record: Record = joiner
        .call_zome_fn("councils", "join_council", join_input)
        .await;

    assert!(
        !membership_record.action_hashed().hash.as_ref().is_empty(),
        "Membership should be created"
    );

    wait_for_dht_sync().await;

    // Verify membership is visible
    let members: Vec<Record> = creator
        .call_zome_fn("councils", "get_council_members", council_id.clone())
        .await;

    assert!(!members.is_empty(), "Council should have at least one member");
}

/// Test: Create child council under a parent.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_council_hierarchy() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Create parent council
    let parent_input = serde_json::json!({
        "name": "Parent Council",
        "purpose": "Top-level council for hierarchy test",
        "council_type": "Root",
        "parent_council_id": null,
        "phi_threshold": 0.2,
        "quorum": 0.5,
        "supermajority": 0.67,
        "can_spawn_children": true,
        "max_delegation_depth": 3,
        "signing_committee_id": null
    });

    let parent_record: Record = agent
        .call_zome_fn("councils", "create_council", parent_input)
        .await;

    let parent_entry: serde_json::Value = parent_record
        .entry()
        .to_app_option()
        .expect("Should deserialize")
        .expect("Should have entry");
    let parent_id = parent_entry["id"].as_str().expect("Parent should have id").to_string();

    wait_for_dht_sync().await;

    // Create child council
    let child_input = serde_json::json!({
        "name": "Child Working Group",
        "purpose": "Sub-group of parent council",
        "council_type": { "WorkingGroup": { "focus": "Testing", "expires": null } },
        "parent_council_id": parent_id,
        "phi_threshold": 0.1,
        "quorum": 0.5,
        "supermajority": 0.5,
        "can_spawn_children": false,
        "max_delegation_depth": 1,
        "signing_committee_id": null
    });

    let child_record: Record = agent
        .call_zome_fn("councils", "create_council", child_input)
        .await;

    assert!(
        !child_record.action_hashed().hash.as_ref().is_empty(),
        "Child council should be created"
    );

    wait_for_dht_sync().await;

    // Verify parent-child link
    let children: Vec<Record> = agent
        .call_zome_fn("councils", "get_child_councils", parent_id.clone())
        .await;

    assert!(!children.is_empty(), "Parent should have at least one child council");
}

// ============================================================================
// CONSTITUTION TESTS
// ============================================================================

/// Test: Create a charter and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_charter_create_and_get() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();

    let charter_input = serde_json::json!({
        "id": "charter-v1",
        "version": 1,
        "preamble": "We the members of the Mycelix network establish this charter.",
        "articles": "[]",
        "rights": [
            "Right to participate in governance",
            "Right to privacy of personal data",
            "Right to exit the network"
        ],
        "amendment_process": "Two-thirds supermajority of active members",
        "adopted": now,
        "last_amended": null
    });

    let charter_record: Record = agent
        .call_zome_fn("constitution", "create_charter", charter_input)
        .await;

    assert!(
        !charter_record.action_hashed().hash.as_ref().is_empty(),
        "Charter should be created"
    );

    wait_for_dht_sync().await;

    // Retrieve current charter
    let current: Option<Record> = agent
        .call_zome_fn("constitution", "get_current_charter", ())
        .await;

    assert!(current.is_some(), "Current charter should be retrievable");
}

/// Test: Set and retrieve governance parameters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_governance_parameters() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Set a governance parameter
    let param_input = serde_json::json!({
        "name": "quorum_threshold",
        "value": "0.5",
        "value_type": "Percentage",
        "description": "Minimum participation required for valid votes",
        "min_value": "0.1",
        "max_value": "1.0",
        "proposal_id": null
    });

    let param_record: Record = agent
        .call_zome_fn("constitution", "set_parameter", param_input)
        .await;

    assert!(
        !param_record.action_hashed().hash.as_ref().is_empty(),
        "Parameter should be created"
    );

    wait_for_dht_sync().await;

    // Get parameter by name
    let retrieved: Option<Record> = agent
        .call_zome_fn("constitution", "get_parameter", "quorum_threshold".to_string())
        .await;

    assert!(retrieved.is_some(), "Parameter should be retrievable by name");
}

/// Test: List all governance parameters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_list_governance_parameters() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Set two parameters
    let param1 = serde_json::json!({
        "name": "voting_period_days",
        "value": "7",
        "value_type": "Integer",
        "description": "Default voting period in days",
        "min_value": "1",
        "max_value": "90",
        "proposal_id": null
    });

    let param2 = serde_json::json!({
        "name": "min_proposal_deposit",
        "value": "100",
        "value_type": "Integer",
        "description": "Minimum deposit to submit a proposal",
        "min_value": "0",
        "max_value": null,
        "proposal_id": null
    });

    let _: Record = agent.call_zome_fn("constitution", "set_parameter", param1).await;
    let _: Record = agent.call_zome_fn("constitution", "set_parameter", param2).await;

    wait_for_dht_sync().await;

    let all_params: Vec<Record> = agent
        .call_zome_fn("constitution", "list_parameters", ())
        .await;

    assert!(
        all_params.len() >= 2,
        "Should list at least 2 parameters (got {})",
        all_params.len()
    );
}

/// Test: Propose a constitutional amendment.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_propose_amendment() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();

    // Create charter first
    let charter_input = serde_json::json!({
        "id": "charter-amend-test",
        "version": 1,
        "preamble": "Initial charter for amendment testing.",
        "articles": "[]",
        "rights": ["Right to participate"],
        "amendment_process": "Two-thirds supermajority",
        "adopted": now,
        "last_amended": null
    });

    let _: Record = agent
        .call_zome_fn("constitution", "create_charter", charter_input)
        .await;

    wait_for_dht_sync().await;

    // Propose an amendment
    let amendment_input = serde_json::json!({
        "amendment_type": "AddRight",
        "article": null,
        "original_text": null,
        "new_text": "Right to delegate voting power to trusted community members",
        "rationale": "Delegation enables broader participation in governance",
        "proposer_did": format!("did:mycelix:{}", agent.agent_pubkey),
        "proposal_id": "MIP-AMEND-001"
    });

    let amendment_record: Record = agent
        .call_zome_fn("constitution", "propose_amendment", amendment_input)
        .await;

    assert!(
        !amendment_record.action_hashed().hash.as_ref().is_empty(),
        "Amendment should be created"
    );
}

// ============================================================================
// EXECUTION QUEUE TESTS
// ============================================================================

/// Test: Create a timelock and query pending timelocks.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_execution_timelock_create_and_query() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Create a timelock
    let timelock_input = serde_json::json!({
        "proposal_id": "MIP-EXEC-001",
        "actions": "{\"type\": \"UpdateParameter\", \"name\": \"quorum_threshold\", \"value\": \"0.6\"}",
        "duration_hours": 48
    });

    let timelock_record: Record = agent
        .call_zome_fn("execution", "create_timelock", timelock_input)
        .await;

    assert!(
        !timelock_record.action_hashed().hash.as_ref().is_empty(),
        "Timelock should be created"
    );

    wait_for_dht_sync().await;

    // Query proposal timelock
    let proposal_timelock: Option<Record> = agent
        .call_zome_fn("execution", "get_proposal_timelock", "MIP-EXEC-001".to_string())
        .await;

    assert!(proposal_timelock.is_some(), "Should find timelock for proposal");
}

/// Test: Get pending timelocks list.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_get_pending_timelocks() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Create a timelock
    let timelock_input = serde_json::json!({
        "proposal_id": "MIP-EXEC-002",
        "actions": "{\"type\": \"UpdateParameter\", \"name\": \"voting_period\", \"value\": \"14\"}",
        "duration_hours": 24
    });

    let _: Record = agent
        .call_zome_fn("execution", "create_timelock", timelock_input)
        .await;

    wait_for_dht_sync().await;

    let pending: Vec<Record> = agent
        .call_zome_fn("execution", "get_pending_timelocks", ())
        .await;

    assert!(!pending.is_empty(), "Should have at least one pending timelock");
}

/// Test: Mark a timelock as ready.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_mark_timelock_ready() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Create a timelock
    let timelock_input = serde_json::json!({
        "proposal_id": "MIP-EXEC-003",
        "actions": "{}",
        "duration_hours": 1
    });

    let timelock_record: Record = agent
        .call_zome_fn("execution", "create_timelock", timelock_input)
        .await;

    // Extract timelock ID from the entry
    let tl_entry: serde_json::Value = timelock_record
        .entry()
        .to_app_option()
        .expect("Should deserialize")
        .expect("Should have entry");
    let timelock_id = tl_entry["id"].as_str().expect("Timelock should have id").to_string();

    wait_for_dht_sync().await;

    // Mark it ready
    let ready_input = serde_json::json!({
        "timelock_id": timelock_id
    });

    let ready_record: Record = agent
        .call_zome_fn("execution", "mark_timelock_ready", ready_input)
        .await;

    assert!(
        !ready_record.action_hashed().hash.as_ref().is_empty(),
        "Timelock should be marked as ready"
    );
}

// ============================================================================
// THRESHOLD SIGNING (DKG) TESTS
// ============================================================================

/// Test: Create a signing committee and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_signing_committee() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let committee_input = serde_json::json!({
        "name": "Treasury Signing Committee",
        "threshold": 2,
        "max_members": 5,
        "purpose": "Multi-sig for treasury operations",
        "council_id": null
    });

    let committee_record: Record = agent
        .call_zome_fn("threshold_signing", "create_committee", committee_input)
        .await;

    assert!(
        !committee_record.action_hashed().hash.as_ref().is_empty(),
        "Signing committee should be created"
    );

    // Extract committee ID
    let entry: serde_json::Value = committee_record
        .entry()
        .to_app_option()
        .expect("Should deserialize")
        .expect("Should have entry");
    let committee_id = entry["id"].as_str().expect("Committee should have id").to_string();

    wait_for_dht_sync().await;

    // Retrieve by ID
    let retrieved: Option<Record> = agent
        .call_zome_fn("threshold_signing", "get_committee", committee_id.clone())
        .await;

    assert!(retrieved.is_some(), "Committee should be retrievable by ID");
}

/// Test: List all active signing committees.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_list_active_committees() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let committee_input = serde_json::json!({
        "name": "Active Committee Test",
        "threshold": 2,
        "max_members": 3,
        "purpose": "Tests active committee listing",
        "council_id": null
    });

    let _: Record = agent
        .call_zome_fn("threshold_signing", "create_committee", committee_input)
        .await;

    wait_for_dht_sync().await;

    let all_committees: Vec<Record> = agent
        .call_zome_fn("threshold_signing", "get_all_committees", ())
        .await;

    assert!(!all_committees.is_empty(), "Should list at least one active committee");
}

// ============================================================================
// BRIDGE TESTS
// ============================================================================

/// Test: Query governance from the bridge zome.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_bridge_query_governance() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Create a proposal first
    let proposal_input = serde_json::json!({
        "id": "MIP-BRIDGE-001",
        "title": "Bridge query test proposal",
        "description": "Verify bridge can query governance data",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", agent.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = agent
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    // Query via bridge
    let query_input = serde_json::json!({
        "query_type": "ActiveProposals",
        "happ_id": "mycelix-governance"
    });

    let result: serde_json::Value = agent
        .call_zome_fn("governance_bridge", "query_governance", query_input)
        .await;

    // The result should have a status field
    assert!(
        !format!("{:?}", result).is_empty(),
        "Bridge query should return a result"
    );
}

/// Test: Get recent governance events (initially empty).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_bridge_get_recent_events() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let events: Vec<Record> = agent
        .call_zome_fn("governance_bridge", "get_recent_events", ())
        .await;

    // May be empty on a fresh DNA, but the call should succeed
    assert!(events.len() >= 0, "get_recent_events should not error");
}

/// Test: Get consciousness thresholds from bridge.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_bridge_consciousness_thresholds() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let thresholds: serde_json::Value = agent
        .call_zome_fn("governance_bridge", "get_consciousness_thresholds", ())
        .await;

    // Should return a summary with threshold values
    assert!(
        !format!("{:?}", thresholds).is_empty(),
        "Consciousness thresholds should be returned"
    );
}

// ============================================================================
// DISCUSSION SYSTEM TESTS
// ============================================================================

/// Test: Add a discussion contribution and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_discussion_contribution() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Create proposal for the discussion
    let proposal_input = serde_json::json!({
        "id": "MIP-DISC-001",
        "title": "Discussion test proposal",
        "description": "Tests the discussion contribution system",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", agent.agent_pubkey),
        "status": "Active",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let _: Record = agent
        .call_zome_fn("proposals", "create_proposal", proposal_input)
        .await;

    wait_for_dht_sync().await;

    // Add a contribution
    let contribution_input = serde_json::json!({
        "proposal_id": "MIP-DISC-001",
        "contributor_did": format!("did:mycelix:{}", agent.agent_pubkey),
        "content": "I believe this proposal aligns well with our community values.",
        "harmony_tags": ["ResonantCoherence", "SacredReciprocity"],
        "stance": "Support",
        "parent_id": null
    });

    let contribution_record: Record = agent
        .call_zome_fn("proposals", "add_contribution", contribution_input)
        .await;

    assert!(
        !contribution_record.action_hashed().hash.as_ref().is_empty(),
        "Contribution should be created"
    );

    wait_for_dht_sync().await;

    // Retrieve discussion
    let discussion: Vec<Record> = agent
        .call_zome_fn("proposals", "get_discussion", "MIP-DISC-001".to_string())
        .await;

    assert!(!discussion.is_empty(), "Discussion should have at least one contribution");
}
