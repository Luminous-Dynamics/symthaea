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
    Cooking,
    Cleaning,
    Childcare,
    Shopping,
    Medical,
    Emotional,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Recurrence {
    Daily,
    Weekly,
    Biweekly,
    Monthly,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCareScheduleInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub care_type: CareType,
    pub assignee: AgentPubKey,
    pub recurrence: Recurrence,
    pub starts_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompleteTaskInput {
    pub schedule_hash: ActionHash,
    pub notes: Option<String>,
}

// ============================================================================
// Mirror types — gratitude
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum GratitudeType {
    Appreciation,
    Thanks,
    Recognition,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExpressGratitudeInput {
    pub hearth_hash: ActionHash,
    pub recipient: AgentPubKey,
    pub gratitude_type: GratitudeType,
    pub message: String,
}

// ============================================================================
// Mirror types — rhythms
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RhythmType {
    Meal,
    Bedtime,
    Gathering,
    Celebration,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateRhythmInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub rhythm_type: RhythmType,
    pub recurrence: Recurrence,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogOccurrenceInput {
    pub rhythm_hash: ActionHash,
    pub notes: Option<String>,
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
        title: "Cook dinner".to_string(),
        care_type: CareType::Cooking,
        assignee: alice.agent_pubkey().clone(),
        recurrence: Recurrence::Daily,
        starts_at: Timestamp::now(),
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
        notes: Some("Made pasta tonight".to_string()),
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

    // 4. Alice expresses gratitude to Bob
    let gratitude_input = ExpressGratitudeInput {
        hearth_hash: hearth_hash.clone(),
        recipient: bob_agent,
        gratitude_type: GratitudeType::Appreciation,
        message: "Thank you for being part of this family".to_string(),
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
        rhythm_type: RhythmType::Meal,
        recurrence: Recurrence::Weekly,
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
        notes: Some("Grandma's famous lasagna".to_string()),
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
