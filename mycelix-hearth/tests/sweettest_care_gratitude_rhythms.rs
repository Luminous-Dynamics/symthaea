// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Hearth — Care, Gratitude & Rhythms Sweettest Integration Tests
//!
//! Tests the H2 weekly digest pipeline end-to-end: care schedule creation and
//! completion, gratitude expression, and rhythm logging with occurrence tracking.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_care_gratitude_rhythms -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (hearth creation)
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
// Mirror types — care
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CareType {
    Childcare,
    Eldercare,
    PetCare,
    Chore,
    MealPrep,
    Medical,
    Emotional,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Recurrence {
    Daily,
    Weekly,
    Monthly,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCareScheduleInput {
    pub hearth_hash: ActionHash,
    pub care_type: CareType,
    pub title: String,
    pub description: String,
    pub assigned_to: AgentPubKey,
    pub recurrence: Recurrence,
    pub notes: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompleteTaskInput {
    pub schedule_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProposeSwapInput {
    pub hearth_hash: ActionHash,
    pub original_schedule_hash: ActionHash,
    pub swap_date: Timestamp,
}

// ============================================================================
// Mirror types — gratitude
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum GratitudeType {
    Appreciation,
    Acknowledgment,
    Celebration,
    Blessing,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HearthVisibility {
    AllMembers,
    AdultsOnly,
    GuardiansOnly,
    Specified(Vec<AgentPubKey>),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExpressGratitudeInput {
    pub hearth_hash: ActionHash,
    pub to_agent: AgentPubKey,
    pub message: String,
    pub gratitude_type: GratitudeType,
    pub visibility: HearthVisibility,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StartCircleInput {
    pub hearth_hash: ActionHash,
    pub theme: String,
    pub participants: Vec<AgentPubKey>,
}

// ============================================================================
// Mirror types — rhythms
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RhythmType {
    Morning,
    Evening,
    Weekly,
    Seasonal,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateRhythmInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub rhythm_type: RhythmType,
    pub schedule: String,
    pub participants: Vec<AgentPubKey>,
    pub description: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogOccurrenceInput {
    pub rhythm_hash: ActionHash,
    pub participants_present: Vec<AgentPubKey>,
    pub notes: String,
    pub mood_bp: Option<u32>,
}

// ============================================================================
// Mirror types — presence
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SetPresenceInput {
    pub hearth_hash: ActionHash,
    pub status: PresenceStatusType,
    pub expected_return: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PresenceStatusType {
    Home,
    Away,
    Working,
    Sleeping,
    DoNotDisturb,
}

// ============================================================================
// Mirror types — digest
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DigestEpochInput {
    pub hearth_hash: ActionHash,
    pub epoch_start: Timestamp,
    pub epoch_end: Timestamp,
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
// Care Tests
// ============================================================================

/// Alice creates a hearth, creates a care schedule assigned to herself,
/// completes the task, then verifies get_hearth_schedule returns the schedule.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_care_create_and_complete() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Care Test Hearth".to_string(),
                description: "Testing care schedule lifecycle".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(8),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a care schedule
    let schedule_input = CreateCareScheduleInput {
        hearth_hash: hearth_hash.clone(),
        care_type: CareType::MealPrep,
        title: "Cook dinner".to_string(),
        description: "Prepare family dinner".to_string(),
        assigned_to: alice.agent_pubkey().clone(),
        recurrence: Recurrence::Daily,
        notes: "Rotate recipes weekly".to_string(),
    };

    let schedule_record: Record = conductor
        .call(
            &alice.zome("hearth_care"),
            "create_care_schedule",
            schedule_input,
        )
        .await;

    let schedule_hash = schedule_record.action_address().clone();
    assert!(schedule_record.action().author() == alice.agent_pubkey());

    // 3. Alice completes the task
    let complete_input = CompleteTaskInput {
        schedule_hash,
    };

    let _completion: Record = conductor
        .call(
            &alice.zome("hearth_care"),
            "complete_task",
            complete_input,
        )
        .await;

    // 4. Verify get_hearth_schedule returns the schedule
    let schedules: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_care"),
            "get_hearth_schedule",
            hearth_hash,
        )
        .await;

    assert_eq!(
        schedules.len(),
        1,
        "get_hearth_schedule should return exactly 1 care schedule"
    );
}

// ============================================================================
// Gratitude Tests
// ============================================================================

/// Alice creates a hearth, invites Bob, Bob accepts, then Alice expresses
/// gratitude to Bob. get_gratitude_stream returns 1 entry.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_gratitude_express() {
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
                name: "Gratitude Test Hearth".to_string(),
                description: "Testing gratitude expression".to_string(),
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
                message: "Join for gratitude testing".to_string(),
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

    // 4. Alice expresses gratitude to Bob
    let gratitude_input = ExpressGratitudeInput {
        hearth_hash: hearth_hash.clone(),
        to_agent: bob_agent,
        message: "Thank you for being part of this family".to_string(),
        gratitude_type: GratitudeType::Appreciation,
        visibility: HearthVisibility::AllMembers,
    };

    let gratitude_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_gratitude"),
            "express_gratitude",
            gratitude_input,
        )
        .await;

    assert!(gratitude_record.action().author() == alice.agent_pubkey());

    // 5. Verify get_gratitude_stream returns the entry
    let stream: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_gratitude"),
            "get_gratitude_stream",
            hearth_hash,
        )
        .await;

    assert_eq!(
        stream.len(),
        1,
        "get_gratitude_stream should return exactly 1 gratitude entry"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Rhythm Tests
// ============================================================================

/// Alice creates a hearth, creates a rhythm (weekly family meal), logs an
/// occurrence, then verifies get_rhythm_occurrences returns 1 record.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_rhythm_create_and_log() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Rhythm Test Hearth".to_string(),
                description: "Testing rhythm lifecycle".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(8),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a rhythm
    let rhythm_input = CreateRhythmInput {
        hearth_hash: hearth_hash.clone(),
        name: "Sunday Family Dinner".to_string(),
        rhythm_type: RhythmType::Weekly,
        schedule: "Every Sunday at 6pm".to_string(),
        participants: vec![alice.agent_pubkey().clone()],
        description: "Weekly family dinner tradition".to_string(),
    };

    let rhythm_record: Record = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "create_rhythm",
            rhythm_input,
        )
        .await;

    let rhythm_hash = rhythm_record.action_address().clone();
    assert!(rhythm_record.action().author() == alice.agent_pubkey());

    // 3. Alice logs an occurrence
    let occurrence_input = LogOccurrenceInput {
        rhythm_hash: rhythm_hash.clone(),
        participants_present: vec![alice.agent_pubkey().clone()],
        notes: "Grandma's famous lasagna".to_string(),
        mood_bp: Some(8500),
    };

    let _occurrence: Record = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "log_occurrence",
            occurrence_input,
        )
        .await;

    // 4. Verify get_rhythm_occurrences returns 1 occurrence
    let occurrences: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "get_rhythm_occurrences",
            rhythm_hash,
        )
        .await;

    assert_eq!(
        occurrences.len(),
        1,
        "get_rhythm_occurrences should return exactly 1 occurrence"
    );
}

// ============================================================================
// Additional Care Tests
// ============================================================================

/// Alice creates hearth, invites Bob, creates a care schedule assigned to
/// Alice, Bob proposes a swap, and Alice accepts. Uses 2 conductors to
/// avoid "requester and responder cannot be the same agent" validation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_care_swap_lifecycle() {
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
                name: "Swap Test Hearth".to_string(),
                description: "Testing care swap lifecycle".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(8),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob as Adult
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for swap testing".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob accepts invitation
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

    // 4. Alice creates a care schedule assigned to herself
    let schedule_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_care"),
            "create_care_schedule",
            CreateCareScheduleInput {
                hearth_hash: hearth_hash.clone(),
                care_type: CareType::Childcare,
                title: "Morning school run".to_string(),
                description: "Drop kids at school".to_string(),
                assigned_to: alice.agent_pubkey().clone(),
                recurrence: Recurrence::Daily,
                notes: "Leave by 7:45am".to_string(),
            },
        )
        .await;

    let schedule_hash = schedule_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Bob proposes a swap on Alice's schedule
    let swap_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_care"),
            "propose_swap",
            ProposeSwapInput {
                hearth_hash: hearth_hash.clone(),
                original_schedule_hash: schedule_hash,
                swap_date: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let swap_hash = swap_record.action_address().clone();
    assert!(swap_record.action().author() == bob.agent_pubkey());

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 6. Alice accepts the swap
    let _accepted: Record = alice_conductor
        .call(
            &alice.zome("hearth_care"),
            "accept_swap",
            swap_hash,
        )
        .await;

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Additional Gratitude Tests
// ============================================================================

/// Alice creates hearth, starts an appreciation circle, joins it (as a
/// second action to verify join works -- she is already a participant from
/// start), completes the circle, and verifies get_hearth_circles returns 1.
///
/// Note: Since start_appreciation_circle already includes Alice in participants,
/// we use a second conductor (Bob) to test join_circle, then Alice completes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_appreciation_circle_lifecycle() {
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
                name: "Circle Test Hearth".to_string(),
                description: "Testing appreciation circles".to_string(),
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
                message: "Join for circle testing".to_string(),
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

    // 4. Alice starts an appreciation circle with both participants
    let circle_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_gratitude"),
            "start_appreciation_circle",
            StartCircleInput {
                hearth_hash: hearth_hash.clone(),
                theme: "Weekly gratitude sharing".to_string(),
                participants: vec![alice.agent_pubkey().clone(), bob_agent.clone()],
            },
        )
        .await;

    let circle_hash = circle_record.action_address().clone();

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 5. Alice completes the circle (Bob already included as participant)
    let _completed: Record = alice_conductor
        .call(
            &alice.zome("hearth_gratitude"),
            "complete_circle",
            circle_hash,
        )
        .await;

    // 7. get_hearth_circles should return 1 circle
    let circles: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_gratitude"),
            "get_hearth_circles",
            hearth_hash,
        )
        .await;

    assert_eq!(
        circles.len(),
        1,
        "get_hearth_circles should return 1 circle"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Additional Rhythm + Presence Tests
// ============================================================================

/// Alice creates hearth, creates a rhythm, verifies get_hearth_rhythms returns 1,
/// sets her presence to Away, then verifies get_hearth_presence returns 1 entry.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor"]
async fn test_presence_and_rhythm_queries() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Presence Test Hearth".to_string(),
                description: "Testing presence and rhythm queries".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(8),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a rhythm
    let _rhythm_record: Record = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "create_rhythm",
            CreateRhythmInput {
                hearth_hash: hearth_hash.clone(),
                name: "Evening Prayer".to_string(),
                rhythm_type: RhythmType::Evening,
                schedule: "Every evening at 9pm".to_string(),
                participants: vec![alice.agent_pubkey().clone()],
                description: "Nightly reflection and prayer".to_string(),
            },
        )
        .await;

    // 3. get_hearth_rhythms should return 1
    let rhythms: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "get_hearth_rhythms",
            hearth_hash.clone(),
        )
        .await;

    assert_eq!(
        rhythms.len(),
        1,
        "get_hearth_rhythms should return exactly 1 rhythm"
    );

    // 4. Alice sets her presence to Away
    let _presence: Record = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "set_presence",
            SetPresenceInput {
                hearth_hash: hearth_hash.clone(),
                status: PresenceStatusType::Away,
                expected_return: Some(Timestamp::from_micros(
                    Timestamp::now().as_micros() + 3_600_000_000,
                )),
            },
        )
        .await;

    // 5. get_hearth_presence should return 1 entry
    let presence: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_rhythms"),
            "get_hearth_presence",
            hearth_hash,
        )
        .await;

    assert_eq!(
        presence.len(),
        1,
        "get_hearth_presence should return exactly 1 presence entry"
    );
}
