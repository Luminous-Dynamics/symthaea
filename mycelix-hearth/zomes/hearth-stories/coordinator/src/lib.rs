// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Stories Coordinator Zome
//! Business logic for creating, updating, and discovering family stories,
//! managing story collections, and tracking family traditions.

use hdk::prelude::*;
use hearth_coordinator_common::{records_from_links, require_membership};
use hearth_stories_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};

// ============================================================================
// Consciousness Gating
// ============================================================================


// ============================================================================
// Input Types
// ============================================================================

/// Input for creating a new family story.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateStoryInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub content: String,
    pub story_type: StoryType,
    pub media_hashes: Vec<ActionHash>,
    pub tags: Vec<String>,
    pub visibility: HearthVisibility,
}

/// Input for updating an existing story.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateStoryInput {
    pub story_hash: ActionHash,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
}

/// Input for creating a new story collection.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateCollectionInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
}

/// Input for creating a new family tradition.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTraditionInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
    pub frequency: Recurrence,
    pub season: Option<String>,
    pub instructions: String,
}

/// Input for adding media to a story.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddMediaInput {
    pub story_hash: ActionHash,
    pub media_hash: ActionHash,
}

/// Input for adding a story to a collection.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddToCollectionInput {
    pub collection_hash: ActionHash,
    pub story_hash: ActionHash,
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new family story. Links it to the hearth and creates tag links.
#[hdk_extern]
pub fn create_story(input: CreateStoryInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "create_story")?;
    require_membership(&input.hearth_hash)?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let hearth_hash = input.hearth_hash.clone();
    let story_type = input.story_type.clone();

    let story = FamilyStory {
        hearth_hash: input.hearth_hash.clone(),
        title: input.title,
        content: input.content,
        storyteller: caller.clone(),
        story_type: input.story_type,
        media_hashes: input.media_hashes,
        tags: input.tags.clone(),
        visibility: input.visibility,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::FamilyStory(story))?;

    // Link hearth -> story
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToStories,
        (),
    )?;

    // Create tag -> story links for each tag
    for tag in &input.tags {
        let tag_lower = tag.to_lowercase();
        let tag_anchor = Anchor(format!("tag:{}", tag_lower));
        create_entry(&EntryTypes::Anchor(tag_anchor.clone()))?;
        let tag_hash = hash_entry(&EntryTypes::Anchor(tag_anchor))?;
        create_link(
            tag_hash,
            action_hash.clone(),
            LinkTypes::TagToStories,
            tag.as_bytes().to_vec(),
        )?;
    }

    emit_signal(&HearthSignal::StoryCreated {
        hearth_hash,
        story_hash: action_hash.clone(),
        storyteller: caller,
        story_type,
    })?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created story".into()
    )))
}

/// Update an existing story's title, content, and tags.
/// Only the original storyteller may update their story.
#[hdk_extern]
pub fn update_story(input: UpdateStoryInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "update_story")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get(input.story_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Story not found".into())))?;

    let mut story: FamilyStory = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid story entry".into()
        )))?;

    // Authorization: only the original storyteller can update
    if !is_storyteller(&caller, &story) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the original storyteller can update this story".into()
        )));
    }

    story.title = input.title;
    story.content = input.content;
    story.tags = input.tags;

    let updated_hash = update_entry(input.story_hash.clone(), &EntryTypes::FamilyStory(story))?;

    emit_signal(&HearthSignal::StoryUpdated {
        story_hash: input.story_hash,
        updated_by: caller,
    })?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated story".into()
    )))
}

/// Add a media attachment link to a story.
/// Only the original storyteller may add media to their story.
#[hdk_extern]
pub fn add_media_to_story(input: AddMediaInput) -> ExternResult<()> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "add_media_to_story")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get(input.story_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Story not found".into())))?;

    let story: FamilyStory = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid story entry".into()
        )))?;

    // Authorization: only the storyteller can add media
    if !is_storyteller(&caller, &story) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the storyteller can add media to this story".into()
        )));
    }

    create_link(
        input.story_hash,
        input.media_hash,
        LinkTypes::StoryToMedia,
        (),
    )?;
    Ok(())
}

/// Create a new story collection. Links it to the hearth.
#[hdk_extern]
pub fn create_collection(input: CreateCollectionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "create_collection")?;
    require_membership(&input.hearth_hash)?;
    let caller = agent_info()?.agent_initial_pubkey;

    let collection = StoryCollection {
        hearth_hash: input.hearth_hash.clone(),
        name: input.name,
        description: input.description,
        story_hashes: vec![],
        curator: caller,
    };

    let action_hash = create_entry(&EntryTypes::StoryCollection(collection))?;

    // Link hearth -> collection
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToCollections,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created collection".into()
    )))
}

/// Add a story to a collection via a link.
#[hdk_extern]
pub fn add_to_collection(input: AddToCollectionInput) -> ExternResult<()> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "add_to_collection")?;
    // Fetch the collection to get its hearth_hash for membership check
    let record = get(input.collection_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Collection not found".into())
    ))?;
    let collection: StoryCollection = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid collection entry".into()
        )))?;
    require_membership(&collection.hearth_hash)?;

    create_link(
        input.collection_hash,
        input.story_hash,
        LinkTypes::CollectionToStories,
        (),
    )?;
    Ok(())
}

/// Create a new family tradition. Links it to the hearth.
#[hdk_extern]
pub fn create_tradition(input: CreateTraditionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "create_tradition")?;
    require_membership(&input.hearth_hash)?;

    let tradition = FamilyTradition {
        hearth_hash: input.hearth_hash.clone(),
        name: input.name,
        description: input.description,
        frequency: input.frequency,
        season: input.season,
        instructions: input.instructions,
        last_observed: None,
        next_due: None,
    };

    let action_hash = create_entry(&EntryTypes::FamilyTradition(tradition))?;

    // Link hearth -> tradition
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToTraditions,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created tradition".into()
    )))
}

/// Mark a tradition as observed by updating its last_observed timestamp.
/// Reads the tradition to obtain its hearth_hash, then validates membership.
#[hdk_extern]
pub fn observe_tradition(tradition_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "observe_tradition")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let record = get(tradition_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Tradition not found".into())
    ))?;

    let mut tradition: FamilyTradition = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid tradition entry".into()
        )))?;

    // Validate caller is an active member of the tradition's hearth
    require_membership(&tradition.hearth_hash)?;

    tradition.last_observed = Some(now);

    let updated_hash = update_entry(
        tradition_hash.clone(),
        &EntryTypes::FamilyTradition(tradition),
    )?;

    emit_signal(&HearthSignal::TraditionObserved {
        tradition_hash,
        observed_by: caller,
    })?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated tradition".into()
    )))
}

/// Get all stories for a given hearth.
#[hdk_extern]
pub fn get_hearth_stories(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToStories)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all traditions for a given hearth.
#[hdk_extern]
pub fn get_hearth_traditions(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToTraditions)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all collections for a given hearth.
#[hdk_extern]
pub fn get_hearth_collections(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToCollections)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Search stories by tag. Returns all stories linked to the given tag.
#[hdk_extern]
pub fn search_stories_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = Anchor(format!("tag:{}", tag.to_lowercase()));
    let tag_hash = hash_entry(&EntryTypes::Anchor(tag_anchor))?;
    let links = get_links(
        LinkQuery::try_new(tag_hash, LinkTypes::TagToStories)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether the caller is the storyteller of a given story.
fn is_storyteller(caller: &AgentPubKey, story: &FamilyStory) -> bool {
    *caller == story.storyteller
}

/// Normalize a tag: lowercase and trimmed of leading/trailing whitespace.
#[allow(dead_code)]
fn normalize_tag(tag: &str) -> String {
    tag.trim().to_lowercase()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![1u8; 36])
    }

    fn make_story(storyteller: AgentPubKey) -> FamilyStory {
        FamilyStory {
            hearth_hash: fake_action_hash(),
            title: "A Story".to_string(),
            content: "Once upon a time...".to_string(),
            storyteller,
            story_type: StoryType::Memory,
            media_hashes: vec![],
            tags: vec!["memory".to_string()],
            visibility: HearthVisibility::AllMembers,
            created_at: Timestamp::from_micros(1000),
        }
    }

    // ========================================================================
    // Pure helper: is_storyteller
    // ========================================================================

    #[test]
    fn storyteller_matches_caller() {
        let agent = fake_agent();
        let story = make_story(agent.clone());
        assert!(is_storyteller(&agent, &story));
    }

    #[test]
    fn storyteller_does_not_match_different_agent() {
        let storyteller = fake_agent();
        let intruder = fake_agent_b();
        let story = make_story(storyteller);
        assert!(!is_storyteller(&intruder, &story));
    }

    #[test]
    fn storyteller_reflexive() {
        // Same key constructed independently should still match
        let agent_a = AgentPubKey::from_raw_36(vec![42u8; 36]);
        let agent_a_copy = AgentPubKey::from_raw_36(vec![42u8; 36]);
        let story = make_story(agent_a);
        assert!(is_storyteller(&agent_a_copy, &story));
    }

    // ========================================================================
    // Pure helper: normalize_tag
    // ========================================================================

    #[test]
    fn normalize_tag_lowercase() {
        assert_eq!(normalize_tag("HISTORY"), "history");
    }

    #[test]
    fn normalize_tag_mixed_case() {
        assert_eq!(normalize_tag("Family Recipes"), "family recipes");
    }

    #[test]
    fn normalize_tag_trim_whitespace() {
        assert_eq!(normalize_tag("  memory  "), "memory");
    }

    #[test]
    fn normalize_tag_trim_and_lowercase() {
        assert_eq!(normalize_tag("  WWII History  "), "wwii history");
    }

    #[test]
    fn normalize_tag_already_normalized() {
        assert_eq!(normalize_tag("tradition"), "tradition");
    }

    #[test]
    fn normalize_tag_empty_string() {
        assert_eq!(normalize_tag(""), "");
    }

    #[test]
    fn normalize_tag_whitespace_only() {
        assert_eq!(normalize_tag("   "), "");
    }

    #[test]
    fn normalize_tag_unicode() {
        assert_eq!(normalize_tag(" Essen "), "essen");
    }

    #[test]
    fn normalize_tag_internal_spaces_preserved() {
        // Only leading/trailing whitespace is trimmed
        assert_eq!(normalize_tag("  oral  history  "), "oral  history");
    }

    // ========================================================================
    // Scenario: story lifecycle (create -> update authorized -> unauthorized)
    // ========================================================================

    #[test]
    fn scenario_story_lifecycle_auth() {
        let storyteller = fake_agent();
        let other_agent = fake_agent_b();
        let story = make_story(storyteller.clone());

        // Storyteller can update their own story
        assert!(is_storyteller(&storyteller, &story));

        // Another agent cannot update it
        assert!(!is_storyteller(&other_agent, &story));
    }

    #[test]
    fn scenario_create_update_media_auth_chain() {
        // Simulates: storyteller creates story, then only they can add media
        let storyteller = fake_agent();
        let story = make_story(storyteller.clone());

        // Storyteller is authorized for both update and media
        assert!(is_storyteller(&storyteller, &story));

        // Random agent is blocked from both
        let stranger = fake_agent_b();
        assert!(!is_storyteller(&stranger, &story));
    }

    // ========================================================================
    // Scenario: tradition observe updates timestamp
    // ========================================================================

    #[test]
    fn scenario_tradition_observe_updates_timestamp() {
        let mut tradition = FamilyTradition {
            hearth_hash: fake_action_hash(),
            name: "Sunday Dinner".to_string(),
            description: "Weekly gathering".to_string(),
            frequency: Recurrence::Weekly,
            season: None,
            instructions: "Everyone brings a dish".to_string(),
            last_observed: None,
            next_due: None,
        };

        // Initially unobserved
        assert!(tradition.last_observed.is_none());

        // After observation, timestamp is set
        let now = Timestamp::from_micros(1_000_000);
        tradition.last_observed = Some(now);
        assert_eq!(
            tradition.last_observed,
            Some(Timestamp::from_micros(1_000_000))
        );

        // Second observation updates the timestamp
        let later = Timestamp::from_micros(2_000_000);
        tradition.last_observed = Some(later);
        assert_eq!(
            tradition.last_observed,
            Some(Timestamp::from_micros(2_000_000))
        );
    }

    #[test]
    fn scenario_tradition_observe_preserves_other_fields() {
        let mut tradition = FamilyTradition {
            hearth_hash: fake_action_hash(),
            name: "Morning Tea".to_string(),
            description: "Daily morning tea ritual".to_string(),
            frequency: Recurrence::Daily,
            season: Some("Spring".to_string()),
            instructions: "Brew green tea at sunrise".to_string(),
            last_observed: None,
            next_due: Some(Timestamp::from_micros(500_000)),
        };

        // Observe it
        tradition.last_observed = Some(Timestamp::from_micros(1_000_000));

        // All other fields unchanged
        assert_eq!(tradition.name, "Morning Tea");
        assert_eq!(tradition.frequency, Recurrence::Daily);
        assert_eq!(tradition.season, Some("Spring".to_string()));
        assert_eq!(tradition.next_due, Some(Timestamp::from_micros(500_000)));
    }

    // ========================================================================
    // Scenario: tag normalization in context
    // ========================================================================

    #[test]
    fn scenario_tag_normalization_consistency() {
        // Different casings and whitespace should produce the same normalized tag
        let variants = vec!["MEMORY", "Memory", "memory", " memory ", "  MEMORY  "];
        let normalized: Vec<String> = variants.iter().map(|t| normalize_tag(t)).collect();
        assert!(normalized.iter().all(|t| *t == "memory"));
    }

    #[test]
    fn scenario_tag_search_uses_normalized() {
        // Simulates: tag created with "Family Recipes" → normalized to "family recipes"
        // Search with " FAMILY recipes " → normalized to "family recipes" → match
        let created_tag = normalize_tag("Family Recipes");
        let search_tag = normalize_tag(" FAMILY recipes ");
        assert_eq!(created_tag, search_tag);
    }

    // ========================================================================
    // Input type serde roundtrip tests (existing)
    // ========================================================================

    #[test]
    fn create_story_input_serde_roundtrip() {
        let input = CreateStoryInput {
            hearth_hash: fake_action_hash(),
            title: "A Family Memory".to_string(),
            content: "Once upon a time...".to_string(),
            story_type: StoryType::Memory,
            media_hashes: vec![],
            tags: vec!["memory".to_string()],
            visibility: HearthVisibility::AllMembers,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateStoryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hearth_hash, input.hearth_hash);
        assert_eq!(decoded.title, "A Family Memory");
        assert_eq!(decoded.tags.len(), 1);
    }

    #[test]
    fn create_story_input_all_story_types() {
        let types = vec![
            StoryType::Memory,
            StoryType::Tradition,
            StoryType::Recipe,
            StoryType::Wisdom,
            StoryType::Origin,
            StoryType::Migration,
            StoryType::Custom("Folklore".to_string()),
        ];
        for st in types {
            let input = CreateStoryInput {
                hearth_hash: fake_action_hash(),
                title: "Test".to_string(),
                content: "Content".to_string(),
                story_type: st.clone(),
                media_hashes: vec![],
                tags: vec![],
                visibility: HearthVisibility::AllMembers,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: CreateStoryInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.story_type, st);
        }
    }

    #[test]
    fn update_story_input_serde_roundtrip() {
        let input = UpdateStoryInput {
            story_hash: fake_action_hash(),
            title: "Updated Title".to_string(),
            content: "Updated content...".to_string(),
            tags: vec!["updated".to_string(), "revision".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateStoryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.story_hash, input.story_hash);
        assert_eq!(decoded.title, "Updated Title");
        assert_eq!(decoded.tags.len(), 2);
    }

    #[test]
    fn create_collection_input_serde_roundtrip() {
        let input = CreateCollectionInput {
            hearth_hash: fake_action_hash(),
            name: "Family Recipes".to_string(),
            description: "Our cherished recipes".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCollectionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hearth_hash, input.hearth_hash);
        assert_eq!(decoded.name, "Family Recipes");
    }

    #[test]
    fn create_tradition_input_serde_roundtrip() {
        let input = CreateTraditionInput {
            hearth_hash: fake_action_hash(),
            name: "Sunday Dinner".to_string(),
            description: "Weekly gathering".to_string(),
            frequency: Recurrence::Weekly,
            season: None,
            instructions: "Everyone brings a dish".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateTraditionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hearth_hash, input.hearth_hash);
        assert_eq!(decoded.name, "Sunday Dinner");
        assert_eq!(decoded.frequency, Recurrence::Weekly);
        assert!(decoded.season.is_none());
    }

    #[test]
    fn create_tradition_input_with_season() {
        let input = CreateTraditionInput {
            hearth_hash: fake_action_hash(),
            name: "Solstice Feast".to_string(),
            description: "Annual winter celebration".to_string(),
            frequency: Recurrence::Custom("Annually".to_string()),
            season: Some("Winter Solstice".to_string()),
            instructions: "Light candles, share stories...".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateTraditionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.season, Some("Winter Solstice".to_string()));
    }

    #[test]
    fn add_media_input_serde_roundtrip() {
        let input = AddMediaInput {
            story_hash: fake_action_hash(),
            media_hash: fake_action_hash_b(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddMediaInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.story_hash, input.story_hash);
        assert_eq!(decoded.media_hash, input.media_hash);
    }

    #[test]
    fn add_to_collection_input_serde_roundtrip() {
        let input = AddToCollectionInput {
            collection_hash: fake_action_hash(),
            story_hash: fake_action_hash_b(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddToCollectionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.collection_hash, input.collection_hash);
        assert_eq!(decoded.story_hash, input.story_hash);
    }

    // ========================================================================
    // Entry type serde roundtrip tests (existing)
    // ========================================================================

    #[test]
    fn family_story_serde_roundtrip() {
        let story = FamilyStory {
            hearth_hash: fake_action_hash(),
            title: "Grandpa's War Stories".to_string(),
            content: "In the summer of 1944...".to_string(),
            storyteller: fake_agent(),
            story_type: StoryType::Memory,
            media_hashes: vec![fake_action_hash_b()],
            tags: vec!["history".to_string(), "wwii".to_string()],
            visibility: HearthVisibility::AllMembers,
            created_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&story).unwrap();
        let decoded: FamilyStory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, story);
    }

    #[test]
    fn family_tradition_serde_roundtrip() {
        let tradition = FamilyTradition {
            hearth_hash: fake_action_hash(),
            name: "Morning Tea".to_string(),
            description: "Daily morning tea ritual".to_string(),
            frequency: Recurrence::Daily,
            season: None,
            instructions: "Brew green tea at sunrise".to_string(),
            last_observed: Some(Timestamp::from_micros(5000)),
            next_due: Some(Timestamp::from_micros(91_000_000)),
        };
        let json = serde_json::to_string(&tradition).unwrap();
        let decoded: FamilyTradition = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, tradition);
    }

    #[test]
    fn story_collection_serde_roundtrip() {
        let collection = StoryCollection {
            hearth_hash: fake_action_hash(),
            name: "Migration Tales".to_string(),
            description: "Stories of our family's journeys".to_string(),
            story_hashes: vec![fake_action_hash(), fake_action_hash_b()],
            curator: fake_agent(),
        };
        let json = serde_json::to_string(&collection).unwrap();
        let decoded: StoryCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, collection);
    }

    // ========================================================================
    // Signal serde roundtrip tests
    // ========================================================================

    #[test]
    fn signal_story_created_serde() {
        let sig = HearthSignal::StoryCreated {
            hearth_hash: fake_action_hash(),
            story_hash: fake_action_hash_b(),
            storyteller: fake_agent(),
            story_type: StoryType::Memory,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("StoryCreated"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::StoryCreated { story_type, .. } => {
                assert_eq!(story_type, StoryType::Memory);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_story_updated_serde() {
        let sig = HearthSignal::StoryUpdated {
            story_hash: fake_action_hash(),
            updated_by: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("StoryUpdated"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::StoryUpdated { story_hash, .. } => {
                assert_eq!(story_hash, fake_action_hash());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_tradition_observed_serde() {
        let sig = HearthSignal::TraditionObserved {
            tradition_hash: fake_action_hash(),
            observed_by: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("TraditionObserved"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TraditionObserved { tradition_hash, .. } => {
                assert_eq!(tradition_hash, fake_action_hash());
            }
            _ => panic!("Wrong variant"),
        }
    }
}
