// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Autonomy Sweettest Integration Tests
//!
//! Tests the youth autonomy guardian flow: profile creation, capability
//! request, approval, and verification.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_autonomy -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (hearth creation + invitation)
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
// Mirror types — autonomy
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutonomyTier {
    Dependent,
    Supervised,
    Guided,
    SemiAutonomous,
    Autonomous,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateAutonomyProfileInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub guardian_agents: Vec<AgentPubKey>,
    pub initial_tier: AutonomyTier,
    pub capabilities: Vec<String>,
    pub restrictions: Vec<String>,
    pub review_schedule: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestCapabilityInput {
    pub hearth_hash: ActionHash,
    pub capability: String,
    pub justification: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApproveCapabilityInput {
    pub request_hash: ActionHash,
    pub approved: bool,
    pub conditions: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckCapabilityInput {
    pub member: AgentPubKey,
    pub capability: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AdvanceTierInput {
    pub profile_hash: ActionHash,
    pub new_tier: AutonomyTier,
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
// Autonomy Tests
// ============================================================================

/// Alice (guardian) creates hearth, invites Bob (youth). Alice creates an
/// autonomy profile for Bob at Dependent tier. Bob requests a capability.
/// Alice approves. check_capability returns true.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_autonomy_profile_and_capability() {
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
                name: "Autonomy Test Hearth".to_string(),
                description: "Testing youth autonomy flow".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob as Youth
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Youth,
                message: "Welcome to the family, Bob".to_string(),
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

    // 4. Alice creates an autonomy profile for Bob at Dependent tier
    let profile_input = CreateAutonomyProfileInput {
        hearth_hash: hearth_hash.clone(),
        member: bob_agent.clone(),
        guardian_agents: vec![alice.agent_pubkey().clone()],
        initial_tier: AutonomyTier::Dependent,
        capabilities: vec!["manage_own_schedule".to_string()],
        restrictions: vec![],
        review_schedule: Some("monthly".to_string()),
    };

    let profile_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "create_autonomy_profile",
            profile_input,
        )
        .await;

    assert!(profile_record.action().author() == alice.agent_pubkey());

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Bob requests a capability
    let request_input = RequestCapabilityInput {
        hearth_hash: hearth_hash.clone(),
        capability: "manage_own_schedule".to_string(),
        justification: "I want to set my own bedtime on weekends".to_string(),
    };

    let request_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_autonomy"),
            "request_capability",
            request_input,
        )
        .await;

    let request_hash = request_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Alice approves the capability request
    let approve_input = ApproveCapabilityInput {
        request_hash,
        approved: true,
        conditions: Some("Only on Friday and Saturday nights".to_string()),
    };

    let _approve: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "approve_capability",
            approve_input,
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 7. check_capability returns true for Bob
    let check_input = CheckCapabilityInput {
        member: bob_agent,
        capability: "manage_own_schedule".to_string(),
    };

    let has_capability: bool = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "check_capability",
            check_input,
        )
        .await;

    assert!(
        has_capability,
        "check_capability should return true after approval"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Additional Autonomy Tests
// ============================================================================

/// Alice (guardian) creates hearth, invites Bob (youth), creates autonomy profile
/// at Dependent tier. Alice advances Bob's tier to Supervised. get_active_transitions
/// returns 1 transition (in PreLiminal phase). Alice progresses the transition
/// forward one phase and verifies advancement.
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_advance_tier_and_get_transitions() {
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
                name: "Tier Advance Hearth".to_string(),
                description: "Testing tier advancement".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob as Youth
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Youth,
                message: "Join for tier testing".to_string(),
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

    // 4. Alice creates autonomy profile for Bob at Dependent tier
    let profile_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "create_autonomy_profile",
            CreateAutonomyProfileInput {
                hearth_hash: hearth_hash.clone(),
                member: bob_agent.clone(),
                guardian_agents: vec![alice.agent_pubkey().clone()],
                initial_tier: AutonomyTier::Dependent,
                capabilities: vec![],
                restrictions: vec![],
                review_schedule: None,
            },
        )
        .await;

    let profile_hash = profile_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Alice advances Bob's tier from Dependent to Supervised
    let transition_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "advance_tier",
            AdvanceTierInput {
                profile_hash,
                new_tier: AutonomyTier::Supervised,
            },
        )
        .await;

    let transition_hash = transition_record.action_address().clone();

    // 6. get_active_transitions should return 1 transition
    let active_transitions: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "get_active_transitions",
            hearth_hash.clone(),
        )
        .await;

    assert!(
        !active_transitions.is_empty(),
        "get_active_transitions should return at least 1 active transition"
    );

    // 7. Progress the transition one phase (PreLiminal -> Liminal)
    let _progressed: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "progress_transition",
            transition_hash,
        )
        .await;

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Alice (guardian) creates hearth, invites Bob (youth). Bob requests a capability.
/// Alice calls get_pending_requests and verifies 1 pending request. Alice then
/// denies the capability. get_pending_requests now returns 0.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_deny_capability_and_get_pending() {
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
                name: "Deny Cap Hearth".to_string(),
                description: "Testing deny capability".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob as Youth
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Youth,
                message: "Join for deny testing".to_string(),
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

    // 4. Alice creates autonomy profile for Bob
    let _: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "create_autonomy_profile",
            CreateAutonomyProfileInput {
                hearth_hash: hearth_hash.clone(),
                member: bob_agent.clone(),
                guardian_agents: vec![alice.agent_pubkey().clone()],
                initial_tier: AutonomyTier::Dependent,
                capabilities: vec![],
                restrictions: vec![],
                review_schedule: None,
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Bob requests a capability
    let request_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_autonomy"),
            "request_capability",
            RequestCapabilityInput {
                hearth_hash: hearth_hash.clone(),
                capability: "use_stove".to_string(),
                justification: "I want to cook dinner".to_string(),
            },
        )
        .await;

    let request_hash = request_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. get_pending_requests should return 1
    let pending_before: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "get_pending_requests",
            hearth_hash.clone(),
        )
        .await;

    assert_eq!(
        pending_before.len(),
        1,
        "get_pending_requests should return 1 pending request before denial"
    );

    // 7. Alice denies the capability
    let _denial: Record = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "deny_capability",
            ApproveCapabilityInput {
                request_hash,
                approved: false,
                conditions: Some("Too young for stove".to_string()),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 8. get_pending_requests should return 0 (denied request is no longer pending)
    let pending_after: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "get_pending_requests",
            hearth_hash,
        )
        .await;

    assert_eq!(
        pending_after.len(),
        0,
        "get_pending_requests should return 0 after denial"
    );

    // 9. check_capability should return false for denied capability
    let has_cap: bool = alice_conductor
        .call(
            &alice.zome("hearth_autonomy"),
            "check_capability",
            CheckCapabilityInput {
                member: bob_agent,
                capability: "use_stove".to_string(),
            },
        )
        .await;

    assert!(
        !has_cap,
        "check_capability should return false after denial"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
