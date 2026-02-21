//! Hearth Stories Coordinator Zome
//! Business logic for creating, updating, and discovering family stories,
//! managing story collections, and tracking family traditions.

use hdk::prelude::*;
use hearth_stories_integrity::*;
use hearth_types::*;

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
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let story = FamilyStory {
        hearth_hash: input.hearth_hash.clone(),
        title: input.title,
        content: input.content,
        storyteller: caller,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created story".into()
    )))
}

/// Update an existing story's title, content, and tags.
#[hdk_extern]
pub fn update_story(input: UpdateStoryInput) -> ExternResult<Record> {
    let record = get(input.story_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Story not found".into())))?;

    let mut story: FamilyStory = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid story entry".into()
        )))?;

    story.title = input.title;
    story.content = input.content;
    story.tags = input.tags;

    let updated_hash = update_entry(input.story_hash, &EntryTypes::FamilyStory(story))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated story".into()
    )))
}

/// Add a media attachment link to a story.
#[hdk_extern]
pub fn add_media_to_story(input: AddMediaInput) -> ExternResult<()> {
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
#[hdk_extern]
pub fn observe_tradition(tradition_hash: ActionHash) -> ExternResult<Record> {
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

    tradition.last_observed = Some(now);

    let updated_hash = update_entry(tradition_hash, &EntryTypes::FamilyTradition(tradition))?;

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

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
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

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![1u8; 36])
    }

    // ── CreateStoryInput serde roundtrip ──────────────────────────────

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

    // ── UpdateStoryInput serde roundtrip ──────────────────────────────

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

    // ── CreateCollectionInput serde roundtrip ─────────────────────────

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

    // ── CreateTraditionInput serde roundtrip ──────────────────────────

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

    // ── AddMediaInput serde roundtrip ─────────────────────────────────

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

    // ── AddToCollectionInput serde roundtrip ──────────────────────────

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

    // ── Entry type serde roundtrip tests ──────────────────────────────

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
}
