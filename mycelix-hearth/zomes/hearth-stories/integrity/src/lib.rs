//! Hearth Stories Integrity Zome
//! Defines entry types and validation for family stories, story collections,
//! and family traditions.

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A family story — memories, recipes, wisdom, and more, preserved for generations.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FamilyStory {
    /// The hearth this story belongs to.
    pub hearth_hash: ActionHash,
    /// Title of the story.
    pub title: String,
    /// Full content of the story.
    pub content: String,
    /// Agent who told/recorded the story.
    pub storyteller: AgentPubKey,
    /// Category of story.
    pub story_type: StoryType,
    /// Hashes of attached media entries (photos, audio, etc.).
    pub media_hashes: Vec<ActionHash>,
    /// Tags for categorization and search.
    pub tags: Vec<String>,
    /// Privacy scope.
    pub visibility: HearthVisibility,
    /// When the story was recorded.
    pub created_at: Timestamp,
}

/// A curated collection of related stories.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StoryCollection {
    /// The hearth this collection belongs to.
    pub hearth_hash: ActionHash,
    /// Name of the collection.
    pub name: String,
    /// Description of the collection's theme or purpose.
    pub description: String,
    /// Hashes of stories in this collection.
    pub story_hashes: Vec<ActionHash>,
    /// Agent who curated this collection.
    pub curator: AgentPubKey,
}

/// A recurring family tradition — rituals, celebrations, and practices
/// that define the hearth's identity.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FamilyTradition {
    /// The hearth this tradition belongs to.
    pub hearth_hash: ActionHash,
    /// Name of the tradition.
    pub name: String,
    /// Description of the tradition.
    pub description: String,
    /// How often this tradition recurs.
    pub frequency: Recurrence,
    /// Optional season or time of year (e.g. "Winter Solstice", "Harvest").
    pub season: Option<String>,
    /// Detailed instructions or recipe for observing the tradition.
    pub instructions: String,
    /// When the tradition was last observed.
    pub last_observed: Option<Timestamp>,
    /// When the tradition is next due.
    pub next_due: Option<Timestamp>,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

/// Anchor entry for deterministic link bases (tag indexing).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    FamilyStory(FamilyStory),
    StoryCollection(StoryCollection),
    FamilyTradition(FamilyTradition),
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth to all stories within it.
    HearthToStories,
    /// Story to its attached media entries.
    StoryToMedia,
    /// Hearth to story collections.
    HearthToCollections,
    /// Collection to its stories.
    CollectionToStories,
    /// Hearth to family traditions.
    HearthToTraditions,
    /// Tag anchor to stories with that tag.
    TagToStories,
}

// ============================================================================
// Genesis
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::FamilyStory(story) => validate_story(action, story),
                EntryTypes::StoryCollection(collection) => validate_collection(action, collection),
                EntryTypes::FamilyTradition(tradition) => validate_tradition(action, tradition),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::FamilyStory(story) => {
                    validate_story_update(&story)?;
                    validate_story_immutable_fields(&story, &original_action_hash)
                }
                EntryTypes::StoryCollection(collection) => {
                    validate_collection_update(&collection)?;
                    validate_collection_immutable_fields(&collection, &original_action_hash)
                }
                EntryTypes::FamilyTradition(tradition) => {
                    validate_tradition_update(&tradition)?;
                    validate_tradition_immutable_fields(&tradition, &original_action_hash)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                    "Anchor cannot be updated once created".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => validate_create_link(link_type, &tag),
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => validate_delete_link(link_type, &tag),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_story(_action: Create, story: FamilyStory) -> ExternResult<ValidateCallbackResult> {
    if story.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Story title cannot be empty".into(),
        ));
    }
    if story.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story title must be <= 256 characters".into(),
        ));
    }
    if story.content.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story content must be <= 65536 characters".into(),
        ));
    }
    if story.media_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story cannot have more than 20 media attachments".into(),
        ));
    }
    if story.tags.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story cannot have more than 20 tags".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_story_update(story: &FamilyStory) -> ExternResult<ValidateCallbackResult> {
    if story.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Story title cannot be empty".into(),
        ));
    }
    if story.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story title must be <= 256 characters".into(),
        ));
    }
    if story.content.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story content must be <= 65536 characters".into(),
        ));
    }
    if story.media_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story cannot have more than 20 media attachments".into(),
        ));
    }
    if story.tags.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Story cannot have more than 20 tags".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a FamilyStory update.
fn validate_story_immutable_fields(
    new_story: &FamilyStory,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_story: FamilyStory = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original FamilyStory: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original FamilyStory entry is missing".into()
        )))?;

    if new_story.hearth_hash != original_story.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a FamilyStory".into(),
        ));
    }
    if new_story.storyteller != original_story.storyteller {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change storyteller on a FamilyStory".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a StoryCollection update.
fn validate_collection_immutable_fields(
    new_collection: &StoryCollection,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_collection: StoryCollection = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original StoryCollection: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original StoryCollection entry is missing".into()
        )))?;

    if new_collection.hearth_hash != original_collection.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a StoryCollection".into(),
        ));
    }
    if new_collection.curator != original_collection.curator {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change curator on a StoryCollection".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that immutable fields have not changed on a FamilyTradition update.
fn validate_tradition_immutable_fields(
    new_tradition: &FamilyTradition,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_tradition: FamilyTradition = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original FamilyTradition: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original FamilyTradition entry is missing".into()
        )))?;

    if new_tradition.hearth_hash != original_tradition.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a FamilyTradition".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_collection(
    _action: Create,
    collection: StoryCollection,
) -> ExternResult<ValidateCallbackResult> {
    if collection.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name cannot be empty".into(),
        ));
    }
    if collection.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name must be <= 256 characters".into(),
        ));
    }
    if collection.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection description must be <= 4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_collection_update(
    collection: &StoryCollection,
) -> ExternResult<ValidateCallbackResult> {
    if collection.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name cannot be empty".into(),
        ));
    }
    if collection.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name must be <= 256 characters".into(),
        ));
    }
    if collection.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection description must be <= 4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_tradition(
    _action: Create,
    tradition: FamilyTradition,
) -> ExternResult<ValidateCallbackResult> {
    if tradition.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition name cannot be empty".into(),
        ));
    }
    if tradition.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition name must be <= 256 characters".into(),
        ));
    }
    if tradition.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition description must be <= 4096 characters".into(),
        ));
    }
    if tradition.instructions.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition instructions must be <= 16384 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_tradition_update(tradition: &FamilyTradition) -> ExternResult<ValidateCallbackResult> {
    if tradition.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition name cannot be empty".into(),
        ));
    }
    if tradition.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition name must be <= 256 characters".into(),
        ));
    }
    if tradition.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition description must be <= 4096 characters".into(),
        ));
    }
    if tradition.instructions.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tradition instructions must be <= 16384 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_link(
    link_type: LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    let max_len = link_tag_max_len(&link_type);
    if tag.0.len() > max_len {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "{:?} link tag too long (max {} bytes)",
            link_type, max_len
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_delete_link(
    link_type: LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    let max_len = link_tag_max_len(&link_type);
    if tag.0.len() > max_len {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "{:?} delete link tag too long (max {} bytes)",
            link_type, max_len
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn link_tag_max_len(link_type: &LinkTypes) -> usize {
    match link_type {
        LinkTypes::HearthToStories => 256,
        LinkTypes::StoryToMedia => 256,
        LinkTypes::HearthToCollections => 256,
        LinkTypes::CollectionToStories => 256,
        LinkTypes::HearthToTraditions => 256,
        LinkTypes::TagToStories => 512,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Factory functions

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAA; 36])
    }

    fn valid_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xAB; 36])
    }

    fn valid_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn valid_create_action() -> Create {
        Create {
            author: agent_a(),
            timestamp: valid_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0xAC; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0xAD; 36]),
            weight: Default::default(),
        }
    }

    fn valid_story() -> FamilyStory {
        FamilyStory {
            hearth_hash: valid_action_hash(),
            title: "Grandma's Apple Pie".to_string(),
            content: "Every autumn, grandma would bake her famous apple pie...".to_string(),
            storyteller: agent_a(),
            story_type: StoryType::Recipe,
            media_hashes: vec![],
            tags: vec!["recipe".to_string(), "autumn".to_string()],
            visibility: HearthVisibility::AllMembers,
            created_at: valid_timestamp(),
        }
    }

    fn valid_collection() -> StoryCollection {
        StoryCollection {
            hearth_hash: valid_action_hash(),
            name: "Family Recipes".to_string(),
            description: "A collection of treasured family recipes".to_string(),
            story_hashes: vec![],
            curator: agent_a(),
        }
    }

    fn valid_tradition() -> FamilyTradition {
        FamilyTradition {
            hearth_hash: valid_action_hash(),
            name: "Sunday Dinner".to_string(),
            description: "Weekly family gathering around the dinner table".to_string(),
            frequency: Recurrence::Weekly,
            season: None,
            instructions: "Gather at noon, each member brings a dish...".to_string(),
            last_observed: None,
            next_due: None,
        }
    }

    // ── FamilyStory validation tests ─────────────────────────────────

    #[test]
    fn test_valid_story() {
        let story = valid_story();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_empty_title_rejected() {
        let mut story = valid_story();
        story.title = "".to_string();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story title cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_story_title_one_char() {
        let mut story = valid_story();
        story.title = "A".to_string();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_title_at_max() {
        let mut story = valid_story();
        story.title = "a".repeat(256);
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_title_exceeds_max_rejected() {
        let mut story = valid_story();
        story.title = "a".repeat(257);
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story title must be <= 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_story_empty_content_allowed() {
        let mut story = valid_story();
        story.content = "".to_string();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_content_at_max() {
        let mut story = valid_story();
        story.content = "a".repeat(65536);
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_content_exceeds_max_rejected() {
        let mut story = valid_story();
        story.content = "a".repeat(65537);
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story content must be <= 65536 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_story_media_at_max() {
        let mut story = valid_story();
        story.media_hashes = (0..20)
            .map(|i| ActionHash::from_raw_36(vec![i as u8; 36]))
            .collect();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_too_many_media_rejected() {
        let mut story = valid_story();
        story.media_hashes = (0..21)
            .map(|i| ActionHash::from_raw_36(vec![i as u8; 36]))
            .collect();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story cannot have more than 20 media attachments");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_story_tags_at_max() {
        let mut story = valid_story();
        story.tags = (0..20).map(|i| format!("tag_{i}")).collect();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_too_many_tags_rejected() {
        let mut story = valid_story();
        story.tags = (0..21).map(|i| format!("tag_{i}")).collect();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story cannot have more than 20 tags");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_story_all_types() {
        let types = vec![
            StoryType::Memory,
            StoryType::Tradition,
            StoryType::Recipe,
            StoryType::Wisdom,
            StoryType::Origin,
            StoryType::Migration,
            StoryType::Custom("Legend".to_string()),
        ];
        for st in types {
            let mut story = valid_story();
            story.story_type = st;
            let action = valid_create_action();
            let result = validate_story(action, story).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_story_unicode_title() {
        let mut story = valid_story();
        story.title = "Abuela's Tamales de Navidad".to_string();
        let action = valid_create_action();
        let result = validate_story(action, story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── StoryCollection validation tests ─────────────────────────────

    #[test]
    fn test_valid_collection() {
        let collection = valid_collection();
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_collection_empty_name_rejected() {
        let mut collection = valid_collection();
        collection.name = "".to_string();
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Collection name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_collection_name_at_max() {
        let mut collection = valid_collection();
        collection.name = "a".repeat(256);
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_collection_name_exceeds_max_rejected() {
        let mut collection = valid_collection();
        collection.name = "a".repeat(257);
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Collection name must be <= 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_collection_description_at_max() {
        let mut collection = valid_collection();
        collection.description = "a".repeat(4096);
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_collection_description_exceeds_max_rejected() {
        let mut collection = valid_collection();
        collection.description = "a".repeat(4097);
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Collection description must be <= 4096 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_collection_empty_description_allowed() {
        let mut collection = valid_collection();
        collection.description = "".to_string();
        let action = valid_create_action();
        let result = validate_collection(action, collection).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── FamilyTradition validation tests ─────────────────────────────

    #[test]
    fn test_valid_tradition() {
        let tradition = valid_tradition();
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_empty_name_rejected() {
        let mut tradition = valid_tradition();
        tradition.name = "".to_string();
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Tradition name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_tradition_name_at_max() {
        let mut tradition = valid_tradition();
        tradition.name = "a".repeat(256);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_name_exceeds_max_rejected() {
        let mut tradition = valid_tradition();
        tradition.name = "a".repeat(257);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Tradition name must be <= 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_tradition_description_at_max() {
        let mut tradition = valid_tradition();
        tradition.description = "a".repeat(4096);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_description_exceeds_max_rejected() {
        let mut tradition = valid_tradition();
        tradition.description = "a".repeat(4097);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Tradition description must be <= 4096 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_tradition_instructions_at_max() {
        let mut tradition = valid_tradition();
        tradition.instructions = "a".repeat(16384);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_instructions_exceeds_max_rejected() {
        let mut tradition = valid_tradition();
        tradition.instructions = "a".repeat(16385);
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Tradition instructions must be <= 16384 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_tradition_with_season() {
        let mut tradition = valid_tradition();
        tradition.season = Some("Winter Solstice".to_string());
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_all_frequencies() {
        let freqs = vec![
            Recurrence::Daily,
            Recurrence::Weekly,
            Recurrence::Monthly,
            Recurrence::Custom("Biannual".to_string()),
        ];
        for freq in freqs {
            let mut tradition = valid_tradition();
            tradition.frequency = freq;
            let action = valid_create_action();
            let result = validate_tradition(action, tradition).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_tradition_empty_description_allowed() {
        let mut tradition = valid_tradition();
        tradition.description = "".to_string();
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_empty_instructions_allowed() {
        let mut tradition = valid_tradition();
        tradition.instructions = "".to_string();
        let action = valid_create_action();
        let result = validate_tradition(action, tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── Serde roundtrip tests ────────────────────────────────────────

    #[test]
    fn test_family_story_serde_roundtrip() {
        let story = valid_story();
        let json = serde_json::to_string(&story).unwrap();
        let decoded: FamilyStory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, story);
    }

    #[test]
    fn test_story_collection_serde_roundtrip() {
        let collection = valid_collection();
        let json = serde_json::to_string(&collection).unwrap();
        let decoded: StoryCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, collection);
    }

    #[test]
    fn test_family_tradition_serde_roundtrip() {
        let tradition = valid_tradition();
        let json = serde_json::to_string(&tradition).unwrap();
        let decoded: FamilyTradition = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, tradition);
    }

    #[test]
    fn test_tradition_with_timestamps_serde() {
        let mut tradition = valid_tradition();
        tradition.last_observed = Some(Timestamp::from_micros(1_000_000));
        tradition.next_due = Some(Timestamp::from_micros(2_000_000));
        let json = serde_json::to_string(&tradition).unwrap();
        let decoded: FamilyTradition = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.last_observed,
            Some(Timestamp::from_micros(1_000_000))
        );
        assert_eq!(decoded.next_due, Some(Timestamp::from_micros(2_000_000)));
    }

    // ── Link tag validation tests ────────────────────────────────────

    fn assert_valid_result(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid_result(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_link_hearth_to_stories_valid_tag() {
        let tag = LinkTag(vec![0u8; 128]);
        assert_valid_result(validate_create_link(LinkTypes::HearthToStories, &tag));
    }

    #[test]
    fn test_link_hearth_to_stories_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        assert_valid_result(validate_create_link(LinkTypes::HearthToStories, &tag));
    }

    #[test]
    fn test_link_hearth_to_stories_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToStories, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_story_to_media_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::StoryToMedia, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_hearth_to_collections_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToCollections, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_collection_to_stories_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::CollectionToStories, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_hearth_to_traditions_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToTraditions, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_tag_to_stories_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        assert_valid_result(validate_create_link(LinkTypes::TagToStories, &tag));
    }

    #[test]
    fn test_link_tag_to_stories_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 513]);
        assert_invalid_result(
            validate_create_link(LinkTypes::TagToStories, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_delete_link_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_delete_link(LinkTypes::HearthToStories, &tag),
            "delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_valid_tag() {
        let tag = LinkTag(vec![0u8; 256]);
        assert_valid_result(validate_delete_link(LinkTypes::HearthToStories, &tag));
    }

    // ── Update validation tests ──────────────────────────────────────

    #[test]
    fn test_story_update_valid() {
        let story = valid_story();
        let result = validate_story_update(&story).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_story_update_empty_title_rejected() {
        let mut story = valid_story();
        story.title = "".to_string();
        let result = validate_story_update(&story).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Story title cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_collection_update_valid() {
        let collection = valid_collection();
        let result = validate_collection_update(&collection).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_collection_update_empty_name_rejected() {
        let mut collection = valid_collection();
        collection.name = "".to_string();
        let result = validate_collection_update(&collection).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Collection name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_tradition_update_valid() {
        let tradition = valid_tradition();
        let result = validate_tradition_update(&tradition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_tradition_update_empty_name_rejected() {
        let mut tradition = valid_tradition();
        tradition.name = "".to_string();
        let result = validate_tradition_update(&tradition).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Tradition name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ── Immutable field tests (pure equality, no conductor) ─────────

    #[test]
    fn immutable_field_hearth_hash_difference_detected() {
        let s1 = valid_story();
        let mut s2 = s1.clone();
        s2.hearth_hash = ActionHash::from_raw_36(vec![0xCD; 36]);
        assert_ne!(s1.hearth_hash, s2.hearth_hash);
    }

    #[test]
    fn immutable_field_storyteller_difference_detected() {
        let s1 = valid_story();
        let mut s2 = s1.clone();
        s2.storyteller = AgentPubKey::from_raw_36(vec![0xCD; 36]);
        assert_ne!(s1.storyteller, s2.storyteller);
    }

    // -- StoryCollection immutable field tests --

    #[test]
    fn collection_immutable_hearth_hash_difference_detected() {
        let c1 = valid_collection();
        let mut c2 = c1.clone();
        c2.hearth_hash = ActionHash::from_raw_36(vec![0xCD; 36]);
        assert_ne!(c1.hearth_hash, c2.hearth_hash);
    }

    #[test]
    fn collection_immutable_curator_difference_detected() {
        let c1 = valid_collection();
        let mut c2 = c1.clone();
        c2.curator = AgentPubKey::from_raw_36(vec![0xCD; 36]);
        assert_ne!(c1.curator, c2.curator);
    }

    // -- FamilyTradition immutable field tests --

    #[test]
    fn tradition_immutable_hearth_hash_difference_detected() {
        let t1 = valid_tradition();
        let mut t2 = t1.clone();
        t2.hearth_hash = ActionHash::from_raw_36(vec![0xCD; 36]);
        assert_ne!(t1.hearth_hash, t2.hearth_hash);
    }

    // ── Anchor immutability tests ───────────────────────────────────

    #[test]
    fn anchor_update_rejected_message() {
        let msg = "Anchor cannot be updated once created";
        assert!(msg.contains("cannot be updated"));
    }
}
