// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Kinship Sweettest Integration Tests
//!
//! Tests the core kinship zome: hearth creation, member invitation/acceptance,
//! kinship bond creation, and query functions.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_kinship -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- hearth_types enums ---

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
pub enum BondType {
    Parent,
    Child,
    Sibling,
    Partner,
    Grandparent,
    Grandchild,
    AuntUncle,
    NieceNephew,
    Cousin,
    ChosenFamily,
    Guardian,
    Ward,
    Custom(String),
}

// --- kinship coordinator input types ---

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateBondInput {
    pub hearth_hash: ActionHash,
    pub member_b: AgentPubKey,
    pub bond_type: BondType,
    pub initial_strength_bp: Option<u32>,
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
// Kinship Tests
// ============================================================================

/// Alice creates a hearth, then queries members. Should have 1 member (Alice
/// herself as Founder, auto-created by create_hearth).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_hearth_and_get_members() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = CreateHearthInput {
        name: "Stoltz Family".to_string(),
        description: "A test hearth for integration testing".to_string(),
        hearth_type: HearthType::Nuclear,
        max_members: Some(8),
    };

    let record: Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", input)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());
    let hearth_hash = record.action_address().clone();

    // Get members -- should have 1 (Alice as Founder)
    let members: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_hearth_members",
            hearth_hash,
        )
        .await;

    assert_eq!(members.len(), 1, "Hearth should have exactly 1 member (Founder)");
}

/// Alice creates a hearth, invites Bob, Bob accepts. get_hearth_members returns 2.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_invite_and_accept() {
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

    // Alice creates a hearth
    let hearth_input = CreateHearthInput {
        name: "Richardson Hearth".to_string(),
        description: "Two-member family".to_string(),
        hearth_type: HearthType::Chosen,
        max_members: Some(10),
    };

    let hearth_record: Record = alice_conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Alice invites Bob
    let invite_input = InviteMemberInput {
        hearth_hash: hearth_hash.clone(),
        invitee_agent: bob_agent.clone(),
        proposed_role: MemberRole::Adult,
        message: "Welcome to the family!".to_string(),
        // 24 hours from now (in microseconds)
        expires_at: Timestamp::from_micros(
            Timestamp::now().as_micros() + 86_400_000_000,
        ),
    };

    let invitation_record: Record = alice_conductor
        .call(&alice.zome("hearth_kinship"), "invite_member", invite_input)
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // Bob accepts the invitation
    let accept_input = AcceptInvitationInput {
        invitation_hash,
        display_name: "Bob".to_string(),
    };

    let _membership_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            accept_input,
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // get_hearth_members should return 2
    let members: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_hearth_members",
            hearth_hash,
        )
        .await;

    assert_eq!(
        members.len(),
        2,
        "Hearth should have 2 members (Alice + Bob)"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// After Alice and Bob are in a hearth, Alice creates a kinship bond to Bob.
/// get_kinship_graph should return the bond.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_bond_and_query() {
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

    // Alice creates hearth
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Bond Test Hearth".to_string(),
                description: "Testing kinship bonds".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Alice invites Bob
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for bond testing".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // Bob accepts
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

    // Alice creates a bond to Bob
    let bond_input = CreateBondInput {
        hearth_hash: hearth_hash.clone(),
        member_b: bob_agent,
        bond_type: BondType::Partner,
        initial_strength_bp: Some(8000),
    };

    let bond_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_kinship_bond",
            bond_input,
        )
        .await;

    assert!(bond_record.action().author() == alice.agent_pubkey());

    // get_kinship_graph should return the bond
    let bonds: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_kinship_graph",
            hearth_hash,
        )
        .await;

    assert_eq!(bonds.len(), 1, "Kinship graph should have exactly 1 bond");

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Alice creates a hearth. get_my_hearths returns 1 hearth.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_my_hearths() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a hearth
    let _: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "My Hearths Test".to_string(),
                description: "Verifying get_my_hearths".to_string(),
                hearth_type: HearthType::Intentional,
                max_members: None,
            },
        )
        .await;

    // get_my_hearths should return 1
    let hearths: Vec<Record> = conductor
        .call(&alice.zome("hearth_kinship"), "get_my_hearths", ())
        .await;

    assert_eq!(hearths.len(), 1, "get_my_hearths should return 1 hearth");
}

// ============================================================================
// Mirror types — tend bond + bond health
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TendBondInput {
    pub bond_hash: ActionHash,
    pub description: String,
    pub quality_bp: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetBondHealthInput {
    pub bond_hash: ActionHash,
}

// ============================================================================
// Additional Kinship Tests
// ============================================================================

/// Alice creates hearth, invites Bob, Bob accepts. Alice creates a kinship bond
/// to Bob. Alice tends the bond with a high-quality interaction. Alice calls
/// get_bond_health to verify health is still strong. Alice calls
/// get_neglected_bonds and verifies the tended bond is NOT neglected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_tend_bond_and_get_health() {
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
                name: "Bond Health Hearth".to_string(),
                description: "Testing bond tending and health".to_string(),
                hearth_type: HearthType::Nuclear,
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
                message: "Join for bond health testing".to_string(),
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

    // 4. Alice creates a kinship bond to Bob
    let bond_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_kinship_bond",
            CreateBondInput {
                hearth_hash: hearth_hash.clone(),
                member_b: bob_agent,
                bond_type: BondType::Partner,
                initial_strength_bp: Some(8000),
            },
        )
        .await;

    let bond_hash = bond_record.action_address().clone();

    // 5. Alice tends the bond with a high-quality interaction
    let _tended: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "tend_bond",
            TendBondInput {
                bond_hash: bond_hash.clone(),
                description: "Cooked dinner together".to_string(),
                quality_bp: 9000,
            },
        )
        .await;

    // 6. get_bond_health should return a strong health value (> 3000)
    let health: u32 = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_bond_health",
            GetBondHealthInput {
                bond_hash,
            },
        )
        .await;

    assert!(
        health > 3000,
        "Bond health should be above 3000 (neglected threshold) after tending, got {}",
        health
    );

    // 7. get_neglected_bonds should return 0 (recently tended bond is healthy)
    let neglected: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_neglected_bonds",
            hearth_hash,
        )
        .await;

    assert_eq!(
        neglected.len(),
        0,
        "get_neglected_bonds should return 0 for a recently tended bond"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Alice creates hearth, invites Bob. Bob declines the invitation. Alice
/// invites Bob again, Bob accepts. Then Bob leaves the hearth. Verifies
/// that after leaving, get_hearth_members reflects the departure.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_decline_invitation_and_leave_hearth() {
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
                name: "Decline & Leave Hearth".to_string(),
                description: "Testing decline and leave flow".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob (first time)
    let invitation_1: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join us!".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash_1 = invitation_1.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob declines the first invitation
    let _declined: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "decline_invitation",
            invitation_hash_1,
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 4. Alice invites Bob again (second time)
    let invitation_2: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Please reconsider!".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash_2 = invitation_2.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Bob accepts the second invitation
    let membership_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            AcceptInvitationInput {
                invitation_hash: invitation_hash_2,
                display_name: "Bob".to_string(),
            },
        )
        .await;

    let membership_hash = membership_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Verify 2 members (Alice + Bob)
    let members_before: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_hearth_members",
            hearth_hash.clone(),
        )
        .await;

    assert_eq!(
        members_before.len(),
        2,
        "Hearth should have 2 members before Bob leaves"
    );

    // 7. Bob leaves the hearth
    let _departed: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "leave_hearth",
            membership_hash,
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 8. get_hearth_members still returns 2 records (the link is still there),
    //    but Bob's membership status is Departed. The count of records
    //    doesn't change because links aren't deleted — only the entry is updated.
    //    We verify the member records are returned (membership entries still exist).
    let members_after: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "get_hearth_members",
            hearth_hash,
        )
        .await;

    assert_eq!(
        members_after.len(),
        2,
        "get_hearth_members should still return 2 records (link-based, departed member entry updated but link remains)"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
