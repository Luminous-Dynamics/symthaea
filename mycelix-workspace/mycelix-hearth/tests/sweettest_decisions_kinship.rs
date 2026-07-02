// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Decisions x Kinship Cross-Zome Sweettest
//!
//! Tests cross-zome behavior: the decisions zome checks membership via the
//! kinship zome. Full decision lifecycle: create hearth, invite/accept,
//! create decision, vote, finalize.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_decisions_kinship -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (same as sweettest_kinship.rs)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HearthType {
    Nuclear,
    Extended,
    Chosen,
    Blended,
    Multigenerational,
    Intentional,
    CoPod,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemberRole {
    Founder,
    Elder,
    Adult,
    Youth,
    Child,
    Guest,
    Ancestor,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MembershipStatus {
    Active,
    Invited,
    Departed,
    Ancestral,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum InvitationStatus {
    Pending,
    Accepted,
    Declined,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateHearthInput {
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub max_members: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InviteMemberInput {
    pub hearth_hash: ActionHash,
    pub invitee_agent: AgentPubKey,
    pub proposed_role: MemberRole,
    pub message: String,
    pub expires_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcceptInvitationInput {
    pub invitation_hash: ActionHash,
    pub display_name: String,
}

// ============================================================================
// Mirror types — decisions
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DecisionType {
    Consensus,
    MajorityVote,
    ElderDecision,
    GuardianDecision,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DecisionStatus {
    Open,
    Closed,
    Finalized,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateDecisionInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub eligible_roles: Vec<MemberRole>,
    pub options: Vec<String>,
    pub deadline: Timestamp,
    pub quorum_bp: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CastVoteInput {
    pub decision_hash: ActionHash,
    pub choice: u32,
    pub reasoning: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FinalizeDecisionInput {
    pub decision_hash: ActionHash,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn hearth_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-hearth/
    path.push("dna");
    path.push("mycelix_hearth.dna");
    path
}

// ============================================================================
// Decision Lifecycle Test
// ============================================================================

/// Full decision lifecycle: Alice creates hearth, invites Bob, Bob accepts.
/// Alice creates a decision (MajorityVote). Both Alice and Bob vote.
/// After the deadline passes, Alice finalizes. get_decision_outcome returns
/// the outcome record.
///
/// Note: This test exercises the cross-zome call from decisions -> kinship
/// for role checking and vote weight calculation. The consciousness gating
/// call (decisions -> bridge -> identity) will be best-effort in the test
/// conductor -- the test validates the kinship-decisions integration path.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_decision_lifecycle() {
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let bob_agent = bob.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Decision Test Hearth".to_string(),
                description: "Testing decision lifecycle".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for decision testing".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob accepts
    let _: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            AcceptInvitationInput {
                invitation_hash,
                display_name: "Bob".to_string(),
            },
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 4. Alice creates a decision with a short deadline (1 second in the past
    //    so finalization can proceed immediately). Using a past deadline means
    //    no votes can be cast (deadline check), so we set deadline to be 10
    //    seconds in the future, cast votes, then wait.
    let deadline_micros = Timestamp::now().as_micros() + 60_000_000; // 60 seconds from now
    let deadline = Timestamp::from_micros(deadline_micros);

    let decision_input = CreateDecisionInput {
        hearth_hash: hearth_hash.clone(),
        title: "Where should we have dinner?".to_string(),
        description: "Family dinner location vote".to_string(),
        decision_type: DecisionType::MajorityVote,
        eligible_roles: vec![MemberRole::Founder, MemberRole::Adult],
        options: vec!["Pizza".to_string(), "Tacos".to_string(), "Sushi".to_string()],
        deadline,
        quorum_bp: None,
    };

    let decision_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "create_decision",
            decision_input,
        )
        .await;

    let decision_hash = decision_record.action_address().clone();

    // 5. Alice votes for Pizza (choice 0)
    let _alice_vote: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "cast_vote",
            CastVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 0,
                reasoning: Some("I love pizza".to_string()),
            },
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Bob votes for Pizza (choice 0)
    let _bob_vote: Record = bob_conductor
        .call(
            &bob.zome("hearth_decisions"),
            "cast_vote",
            CastVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 0,
                reasoning: Some("Pizza sounds great".to_string()),
            },
        )
        .await;

    // 7. Wait for deadline to pass (60s deadline from creation)
    tokio::time::sleep(std::time::Duration::from_secs(50)).await;

    // 8. Alice finalizes
    let outcome_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "finalize_decision",
            FinalizeDecisionInput {
                decision_hash: decision_hash.clone(),
            },
        )
        .await;

    assert!(
        outcome_record.action().author() == alice.agent_pubkey(),
        "Outcome should be authored by Alice"
    );

    // 9. get_decision_outcome should return the outcome
    let outcome: Option<Record> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_decision_outcome",
            decision_hash,
        )
        .await;

    assert!(
        outcome.is_some(),
        "get_decision_outcome should return the finalized outcome"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Mirror types — close/amend decisions
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CloseDecisionInput {
    pub decision_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AmendVoteInput {
    pub decision_hash: ActionHash,
    pub choice: u32,
    pub reasoning: Option<String>,
}

// ============================================================================
// Additional Decision Tests
// ============================================================================

/// Alice creates hearth, invites Bob, creates a decision (MajorityVote).
/// Alice votes for option 0, then amends her vote to option 1.
/// Alice closes the decision before the deadline. Verifies the decision
/// status is Closed (via get_decision returning a record).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_close_and_amend_decision() {
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let bob_agent = bob.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Close/Amend Decision Hearth".to_string(),
                description: "Testing close and amend decision".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for close/amend testing".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob accepts
    let _: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            AcceptInvitationInput {
                invitation_hash,
                display_name: "Bob".to_string(),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 4. Alice creates a decision with a 30-second deadline
    let deadline = Timestamp::from_micros(
        Timestamp::now().as_micros() + 30_000_000,
    );

    let decision_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "create_decision",
            CreateDecisionInput {
                hearth_hash: hearth_hash.clone(),
                title: "Movie night pick".to_string(),
                description: "What movie to watch?".to_string(),
                decision_type: DecisionType::MajorityVote,
                eligible_roles: vec![MemberRole::Founder, MemberRole::Adult],
                options: vec![
                    "Comedy".to_string(),
                    "Drama".to_string(),
                    "Action".to_string(),
                ],
                deadline,
                quorum_bp: None,
            },
        )
        .await;

    let decision_hash = decision_record.action_address().clone();

    // 5. Alice votes for Comedy (choice 0)
    let _alice_vote: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "cast_vote",
            CastVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 0,
                reasoning: Some("I want to laugh".to_string()),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Alice amends her vote to Drama (choice 1)
    let _amended_vote: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "amend_vote",
            AmendVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 1,
                reasoning: Some("Actually I changed my mind, want drama".to_string()),
            },
        )
        .await;

    // 7. Alice closes the decision before deadline
    let closed_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "close_decision",
            CloseDecisionInput {
                decision_hash: decision_hash.clone(),
            },
        )
        .await;

    assert!(
        closed_record.action().author() == alice.agent_pubkey(),
        "Closed decision should be authored by Alice"
    );

    // 8. Verify the decision is now retrievable (closed status)
    let fetched: Option<Record> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_decision",
            decision_hash,
        )
        .await;

    assert!(
        fetched.is_some(),
        "get_decision should return the closed decision"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Alice creates hearth, invites Bob. Alice creates a decision. Both Alice and
/// Bob vote. Alice calls tally_votes, get_decision_votes, get_vote_history,
/// and get_my_pending_votes to verify all query functions work correctly.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_tally_and_query_votes() {
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let bob_agent = bob.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Tally Test Hearth".to_string(),
                description: "Testing tally and vote queries".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for tally testing".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob accepts
    let _: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            AcceptInvitationInput {
                invitation_hash,
                display_name: "Bob".to_string(),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 4. Alice creates a decision with 30-second deadline
    let deadline = Timestamp::from_micros(
        Timestamp::now().as_micros() + 30_000_000,
    );

    let decision_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "create_decision",
            CreateDecisionInput {
                hearth_hash: hearth_hash.clone(),
                title: "Breakfast choice".to_string(),
                description: "What for breakfast?".to_string(),
                decision_type: DecisionType::MajorityVote,
                eligible_roles: vec![MemberRole::Founder, MemberRole::Adult],
                options: vec!["Pancakes".to_string(), "Waffles".to_string()],
                deadline,
                quorum_bp: None,
            },
        )
        .await;

    let decision_hash = decision_record.action_address().clone();

    // 5. Alice votes for Pancakes (choice 0)
    let _alice_vote: Record = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "cast_vote",
            CastVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 0,
                reasoning: Some("Love pancakes".to_string()),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Bob votes for Waffles (choice 1)
    let _bob_vote: Record = bob_conductor
        .call(
            &bob.zome("hearth_decisions"),
            "cast_vote",
            CastVoteInput {
                decision_hash: decision_hash.clone(),
                choice: 1,
                reasoning: Some("Waffles are better".to_string()),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 7. tally_votes should return tallies for both options
    let tallies: Vec<(u32, u32)> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "tally_votes",
            decision_hash.clone(),
        )
        .await;

    assert!(
        !tallies.is_empty(),
        "tally_votes should return at least 1 tally entry"
    );

    // 8. get_decision_votes should return 2 votes
    let votes: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_decision_votes",
            decision_hash.clone(),
        )
        .await;

    assert_eq!(
        votes.len(),
        2,
        "get_decision_votes should return 2 votes (Alice + Bob)"
    );

    // 9. get_vote_history should also return 2 (no amendments)
    let history: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_vote_history",
            decision_hash.clone(),
        )
        .await;

    assert_eq!(
        history.len(),
        2,
        "get_vote_history should return 2 entries (no amendments)"
    );

    // 10. get_my_pending_votes for Alice should return 0 (she already voted)
    let alice_pending: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_my_pending_votes",
            hearth_hash.clone(),
        )
        .await;

    assert_eq!(
        alice_pending.len(),
        0,
        "get_my_pending_votes should return 0 for Alice (already voted)"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
