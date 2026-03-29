// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Milestones Sweettest Integration Tests
//!
//! Tests the milestones zome: recording milestones, beginning and advancing
//! life transitions, guardian authorization, and transition completion.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_milestones -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (needed for hearth creation + invite flow)
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
// Mirror types — milestones
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MilestoneType {
    Birth,
    Birthday,
    FirstStep,
    SchoolStart,
    Graduation,
    Engagement,
    Marriage,
    NewHome,
    Retirement,
    Passing,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TransitionType {
    JoiningHearth,
    LeavingHearth,
    ComingOfAge,
    Retirement,
    Bereavement,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum TransitionPhase {
    PreLiminal,
    Liminal,
    PostLiminal,
    Integrated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordMilestoneInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub milestone_type: MilestoneType,
    pub date: Timestamp,
    pub description: String,
    pub witnesses: Vec<AgentPubKey>,
    pub media_hashes: Vec<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BeginTransitionInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub transition_type: TransitionType,
    pub supporting_members: Vec<AgentPubKey>,
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
// Milestones Tests
// ============================================================================

/// Alice creates a hearth, records a milestone, and queries it via
/// get_family_timeline.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_record_milestone() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let alice_agent = alice.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Milestone Test Hearth".to_string(),
                description: "Testing milestone recording".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice records a milestone
    let milestone_record: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "record_milestone",
            RecordMilestoneInput {
                hearth_hash: hearth_hash.clone(),
                member: alice_agent.clone(),
                milestone_type: MilestoneType::Graduation,
                date: Timestamp::from_micros(
                    Timestamp::now().as_micros(),
                ),
                description: "Graduated from university!".to_string(),
                witnesses: vec![],
                media_hashes: vec![],
            },
        )
        .await;

    assert!(milestone_record.action().author() == alice.agent_pubkey());

    // 3. Query family timeline — should return 1 milestone
    let timeline: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "get_family_timeline",
            hearth_hash,
        )
        .await;

    assert_eq!(
        timeline.len(),
        1,
        "Family timeline should have exactly 1 milestone"
    );

    // 4. Query member milestones — should also return 1
    let member_milestones: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "get_member_milestones",
            alice_agent,
        )
        .await;

    assert_eq!(
        member_milestones.len(),
        1,
        "Member milestones should have exactly 1 milestone"
    );
}

/// Alice creates a hearth, begins a transition (ComingOfAge), and advances
/// it through phases: PreLiminal -> Liminal -> PostLiminal.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_begin_and_advance_transition() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let alice_agent = alice.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Transition Test Hearth".to_string(),
                description: "Testing transition advancement".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice begins a transition for herself (any member can begin)
    let transition_record: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "begin_transition",
            BeginTransitionInput {
                hearth_hash: hearth_hash.clone(),
                member: alice_agent.clone(),
                transition_type: TransitionType::ComingOfAge,
                supporting_members: vec![],
            },
        )
        .await;

    let transition_hash = transition_record.action_address().clone();

    // 3. Alice advances: PreLiminal -> Liminal (Alice is Founder = guardian)
    let advanced_record: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "advance_transition",
            transition_hash.clone(),
        )
        .await;

    // The advance should succeed; the updated record is returned
    assert!(advanced_record.action().author() == alice.agent_pubkey());

    // 4. Advance again: Liminal -> PostLiminal
    // advance_transition takes the original action hash
    let advanced_record_2: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "advance_transition",
            transition_hash.clone(),
        )
        .await;

    assert!(advanced_record_2.action().author() == alice.agent_pubkey());

    // 5. Check active transitions — should still have 1 (not yet completed)
    let active: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "get_active_transitions",
            hearth_hash,
        )
        .await;

    assert!(
        !active.is_empty(),
        "Should have at least 1 active transition"
    );
}

/// Non-guardian agent (Youth role) cannot advance a transition.
/// Only guardians (Founder/Elder/Adult) may advance.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_guardian_required_for_advance() {
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
                name: "Guardian Test Hearth".to_string(),
                description: "Testing guardian-only advance".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob as Youth (non-guardian role)
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Youth,
                message: "Join as Youth for testing".to_string(),
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

    // 4. Alice begins a transition
    let transition_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_milestones"),
            "begin_transition",
            BeginTransitionInput {
                hearth_hash: hearth_hash.clone(),
                member: bob_agent.clone(),
                transition_type: TransitionType::ComingOfAge,
                supporting_members: vec![],
            },
        )
        .await;

    let transition_hash = transition_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Bob (Youth) tries to advance the transition — should fail
    let result: Result<Record, _> = bob_conductor
        .call_fallible(
            &bob.zome("hearth_milestones"),
            "advance_transition",
            transition_hash,
        )
        .await;

    assert!(
        result.is_err(),
        "Non-guardian (Youth) should not be able to advance a transition"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Full transition lifecycle: begin -> advance 3 times (Pre->Lim->Post->Integrated)
/// -> complete. Verifies the transition moves through all phases and
/// the final complete_transition call succeeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_complete_transition() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let alice_agent = alice.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Complete Transition Hearth".to_string(),
                description: "Testing full transition lifecycle".to_string(),
                hearth_type: HearthType::Intentional,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Begin transition
    let transition_record: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "begin_transition",
            BeginTransitionInput {
                hearth_hash: hearth_hash.clone(),
                member: alice_agent.clone(),
                transition_type: TransitionType::JoiningHearth,
                supporting_members: vec![],
            },
        )
        .await;

    let transition_hash = transition_record.action_address().clone();

    // 3. Advance: PreLiminal -> Liminal
    let advanced1: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "advance_transition",
            transition_hash.clone(),
        )
        .await;

    let transition_hash_v2 = advanced1.action_address().clone();

    // 4. Advance: Liminal -> PostLiminal
    let advanced2: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "advance_transition",
            transition_hash_v2,
        )
        .await;

    let transition_hash_v3 = advanced2.action_address().clone();

    // 5. Complete the transition (must be in PostLiminal to complete)
    let completed_record: Record = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "complete_transition",
            transition_hash_v3,
        )
        .await;

    assert!(
        completed_record.action().author() == alice.agent_pubkey(),
        "Completed transition should be authored by Alice"
    );

    // 6. Verify the transition is no longer active (records_from_links now
    // follows update chains via get_latest_record, so completed entries are
    // correctly filtered by get_active_transitions)
    let active: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_milestones"),
            "get_active_transitions",
            hearth_hash,
        )
        .await;

    assert_eq!(
        active.len(),
        0,
        "Completed transition should not appear in active transitions"
    );
}
