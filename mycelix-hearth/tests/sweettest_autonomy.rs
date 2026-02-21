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
    Seedling,
    Sprout,
    Sapling,
    YoungTree,
    FullTree,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateAutonomyProfileInput {
    pub hearth_hash: ActionHash,
    pub youth_agent: AgentPubKey,
    pub initial_tier: AutonomyTier,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestCapabilityInput {
    pub hearth_hash: ActionHash,
    pub capability: String,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApproveCapabilityInput {
    pub request_hash: ActionHash,
    pub conditions: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckCapabilityInput {
    pub agent: AgentPubKey,
    pub capability: String,
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
/// autonomy profile for Bob at Seedling tier. Bob requests a capability.
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
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

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
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // 4. Alice creates an autonomy profile for Bob at Seedling tier
    let profile_input = CreateAutonomyProfileInput {
        hearth_hash: hearth_hash.clone(),
        youth_agent: bob_agent.clone(),
        initial_tier: AutonomyTier::Seedling,
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
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // 5. Bob requests a capability
    let request_input = RequestCapabilityInput {
        hearth_hash: hearth_hash.clone(),
        capability: "manage_own_schedule".to_string(),
        reason: "I want to set my own bedtime on weekends".to_string(),
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
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // 6. Alice approves the capability request
    let approve_input = ApproveCapabilityInput {
        request_hash,
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
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // 7. check_capability returns true for Bob
    let check_input = CheckCapabilityInput {
        agent: bob_agent,
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
