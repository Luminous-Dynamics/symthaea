// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Emergency Sweettest Integration Tests
//!
//! Tests the emergency alert lifecycle: plan creation, alert raising,
//! check-in from another member, and alert resolution.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_emergency -- --ignored --test-threads=2
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
// Mirror types — emergency
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Must match hearth_types::AlertType variant ORDER (positional msgpack)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AlertType {
    Medical,
    Natural,
    Security,
    Missing,
    Fire,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateEmergencyPlanInput {
    pub hearth_hash: ActionHash,
    pub contacts: Vec<EmergencyContact>,
    pub meeting_points: Vec<String>,
    pub medical_info_hashes: Vec<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmergencyContact {
    pub name: String,
    pub phone: String,
    pub relationship: String,
    pub priority_order: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RaiseAlertInput {
    pub hearth_hash: ActionHash,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub location_hint: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SafetyStatus {
    Safe,
    NeedHelp,
    NoResponse,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckInInput {
    pub alert_hash: ActionHash,
    pub status: SafetyStatus,
    pub location_hint: Option<String>,
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
// Emergency Lifecycle Tests
// ============================================================================

/// Full emergency lifecycle: Alice creates hearth + emergency plan, raises an
/// alert, Bob checks in, Alice resolves the alert. get_active_alerts returns 0
/// after resolution.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_lifecycle() {
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
                name: "Emergency Test Hearth".to_string(),
                description: "Testing emergency lifecycle".to_string(),
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
                message: "Join for emergency testing".to_string(),
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

    // 4. Alice creates an emergency plan
    let plan_input = CreateEmergencyPlanInput {
        hearth_hash: hearth_hash.clone(),
        contacts: vec![
            EmergencyContact {
                name: "Fire Department".to_string(),
                phone: "911".to_string(),
                relationship: "Emergency Service".to_string(),
                priority_order: 1,
            },
            EmergencyContact {
                name: "Neighbor Jane".to_string(),
                phone: "555-0123".to_string(),
                relationship: "Neighbor".to_string(),
                priority_order: 2,
            },
        ],
        meeting_points: vec!["Oak tree in backyard".to_string()],
        medical_info_hashes: vec![],
    };

    let plan_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_emergency"),
            "create_emergency_plan",
            plan_input,
        )
        .await;

    assert!(plan_record.action().author() == alice.agent_pubkey());

    // 5. Alice raises an alert
    let alert_input = RaiseAlertInput {
        hearth_hash: hearth_hash.clone(),
        alert_type: AlertType::Fire,
        severity: AlertSeverity::High,
        message: "Smoke detected in kitchen".to_string(),
        location_hint: Some("Kitchen area".to_string()),
    };

    let alert_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_emergency"),
            "raise_alert",
            alert_input,
        )
        .await;

    let alert_hash = alert_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Bob checks in
    let checkin_input = CheckInInput {
        alert_hash: alert_hash.clone(),
        status: SafetyStatus::Safe,
        location_hint: Some("Outside, front yard".to_string()),
    };

    let _checkin: Record = bob_conductor
        .call(
            &bob.zome("hearth_emergency"),
            "check_in",
            checkin_input,
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 7. Alice resolves the alert
    let _resolve: Record = alice_conductor
        .call(
            &alice.zome("hearth_emergency"),
            "resolve_alert",
            alert_hash,
        )
        .await;

    // 8. Verify get_active_alerts returns 0
    let active_alerts: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_emergency"),
            "get_active_alerts",
            hearth_hash,
        )
        .await;

    assert_eq!(
        active_alerts.len(),
        0,
        "get_active_alerts should return 0 after resolution"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
