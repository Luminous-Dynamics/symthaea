//! # Mycelix Hearth — Stories Sweettest Integration Tests
//!
//! Tests the stories zome: creating and querying stories, updating stories,
//! collection management, tradition tracking, and membership authorization.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_stories -- --ignored --test-threads=2
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
// Mirror types — stories
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum StoryType {
    Memory,
    Tradition,
    Recipe,
    Wisdom,
    Origin,
    Migration,
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
pub enum Recurrence {
    Daily,
    Weekly,
    Monthly,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateStoryInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub content: String,
    pub story_type: StoryType,
    pub media_hashes: Vec<ActionHash>,
    pub tags: Vec<String>,
    pub visibility: HearthVisibility,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateStoryInput {
    pub story_hash: ActionHash,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCollectionInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddToCollectionInput {
    pub collection_hash: ActionHash,
    pub story_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateTraditionInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
    pub frequency: Recurrence,
    pub season: Option<String>,
    pub instructions: String,
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
// Stories Tests
// ============================================================================

/// Alice creates a hearth, creates a family story, and queries it back
/// via get_hearth_stories.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_and_query_story() {
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
                name: "Story Test Hearth".to_string(),
                description: "Testing story creation".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a story
    let story_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "create_story",
            CreateStoryInput {
                hearth_hash: hearth_hash.clone(),
                title: "Grandma's Apple Pie".to_string(),
                content: "Every autumn, grandma would bake her famous apple pie with cinnamon and love.".to_string(),
                story_type: StoryType::Recipe,
                media_hashes: vec![],
                tags: vec!["recipe".to_string(), "autumn".to_string()],
                visibility: HearthVisibility::AllMembers,
            },
        )
        .await;

    assert!(story_record.action().author() == alice.agent_pubkey());

    // 3. Query hearth stories — should return 1 story
    let stories: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_stories"),
            "get_hearth_stories",
            hearth_hash,
        )
        .await;

    assert_eq!(
        stories.len(),
        1,
        "Hearth should have exactly 1 story"
    );
}

/// Alice creates a story, then updates its title, content, and tags.
/// The update should succeed (storyteller authorization).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_update_story() {
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
                name: "Update Story Hearth".to_string(),
                description: "Testing story updates".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(5),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a story
    let story_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "create_story",
            CreateStoryInput {
                hearth_hash: hearth_hash.clone(),
                title: "Original Title".to_string(),
                content: "Original content of the story.".to_string(),
                story_type: StoryType::Memory,
                media_hashes: vec![],
                tags: vec!["draft".to_string()],
                visibility: HearthVisibility::AllMembers,
            },
        )
        .await;

    let story_hash = story_record.action_address().clone();

    // 3. Alice updates the story
    let updated_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "update_story",
            UpdateStoryInput {
                story_hash,
                title: "Updated Title: The Great Migration".to_string(),
                content: "Updated content with more details about the family journey.".to_string(),
                tags: vec!["migration".to_string(), "history".to_string()],
            },
        )
        .await;

    assert!(
        updated_record.action().author() == alice.agent_pubkey(),
        "Updated story should be authored by Alice"
    );
}

/// Alice creates a collection, creates a story, and adds the story to the
/// collection. get_hearth_collections should return the collection.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_collection_and_add() {
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
                name: "Collection Test Hearth".to_string(),
                description: "Testing story collections".to_string(),
                hearth_type: HearthType::Extended,
                max_members: Some(20),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a collection
    let collection_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "create_collection",
            CreateCollectionInput {
                hearth_hash: hearth_hash.clone(),
                name: "Family Recipes".to_string(),
                description: "Our cherished family recipes passed down through generations".to_string(),
            },
        )
        .await;

    let collection_hash = collection_record.action_address().clone();

    // 3. Alice creates a story
    let story_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "create_story",
            CreateStoryInput {
                hearth_hash: hearth_hash.clone(),
                title: "Tamales de Navidad".to_string(),
                content: "Every Christmas Eve, the whole family gathers to make tamales.".to_string(),
                story_type: StoryType::Recipe,
                media_hashes: vec![],
                tags: vec!["recipe".to_string(), "christmas".to_string()],
                visibility: HearthVisibility::AllMembers,
            },
        )
        .await;

    let story_hash = story_record.action_address().clone();

    // 4. Alice adds the story to the collection
    let _: () = conductor
        .call(
            &alice.zome("hearth_stories"),
            "add_to_collection",
            AddToCollectionInput {
                collection_hash,
                story_hash,
            },
        )
        .await;

    // 5. Query collections — should return 1
    let collections: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_stories"),
            "get_hearth_collections",
            hearth_hash,
        )
        .await;

    assert_eq!(
        collections.len(),
        1,
        "Hearth should have exactly 1 collection"
    );
}

/// Alice creates a family tradition and queries it back. Then observes it
/// to verify the observe_tradition function works.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_tradition() {
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
                name: "Tradition Test Hearth".to_string(),
                description: "Testing family traditions".to_string(),
                hearth_type: HearthType::Multigenerational,
                max_members: Some(15),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a tradition
    let tradition_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "create_tradition",
            CreateTraditionInput {
                hearth_hash: hearth_hash.clone(),
                name: "Sunday Dinner".to_string(),
                description: "Weekly family gathering around the dinner table".to_string(),
                frequency: Recurrence::Weekly,
                season: None,
                instructions: "Everyone brings a dish. We gather at noon and share stories.".to_string(),
            },
        )
        .await;

    let tradition_hash = tradition_record.action_address().clone();

    // 3. Query traditions — should return 1
    let traditions: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_stories"),
            "get_hearth_traditions",
            hearth_hash,
        )
        .await;

    assert_eq!(
        traditions.len(),
        1,
        "Hearth should have exactly 1 tradition"
    );

    // 4. Alice observes the tradition
    let observed_record: Record = conductor
        .call(
            &alice.zome("hearth_stories"),
            "observe_tradition",
            tradition_hash,
        )
        .await;

    assert!(
        observed_record.action().author() == alice.agent_pubkey(),
        "Observed tradition should be authored by Alice"
    );
}

/// A non-member agent cannot create stories for a hearth they don't
/// belong to. This tests the membership authorization.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_membership_required() {
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

    // 1. Alice creates a hearth (Bob is NOT invited)
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Private Hearth".to_string(),
                description: "Testing non-member rejection for stories".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(5),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 2. Bob tries to create a story — should fail (not a member)
    let result: Result<Record, _> = bob_conductor
        .call_fallible(
            &bob.zome("hearth_stories"),
            "create_story",
            CreateStoryInput {
                hearth_hash,
                title: "Unauthorized Story".to_string(),
                content: "This should be rejected".to_string(),
                story_type: StoryType::Memory,
                media_hashes: vec![],
                tags: vec![],
                visibility: HearthVisibility::AllMembers,
            },
        )
        .await;

    assert!(
        result.is_err(),
        "Non-member should not be able to create a story"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
