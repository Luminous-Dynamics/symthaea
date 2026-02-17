//! Search Integrity Zome - Entry types for search index entries
use hdi::prelude::*;
use std::collections::HashMap;

/// Entity types that can be indexed for search
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum EntityType {
    Listing,
    Transaction,
    Conversation,
    Review,
    Agent,
}

/// Search index entry for an entity
///
/// This stores pre-processed search data to enable fast full-text search
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SearchIndexEntry {
    /// Unique index ID
    pub index_id: String,

    /// Entity type being indexed
    pub entity_type: EntityType,

    /// Original entry hash
    pub entity_hash: ActionHash,

    /// Searchable fields
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub category: String,

    /// Metadata for ranking
    pub creator: AgentPubKey,
    pub creator_matl_score: f64,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,

    /// Filters
    pub price_cents: Option<u64>,
    pub status: String,

    /// Processed terms for search (stemmed, normalized)
    pub processed_terms: Vec<String>,

    /// Term frequency map for TF-IDF ranking
    pub term_frequencies: HashMap<String, u32>,
}

/// Link types for search index
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from entity to its search index
    EntityToIndex,

    /// Links from search term to index entries (for fast lookup)
    TermToIndex,

    /// All search indices anchor
    AllIndices,

    /// Links by entity type
    EntityTypeToIndex,
}

/// Entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    SearchIndexEntry(SearchIndexEntry),
}

/// Validation function for SearchIndexEntry
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::SearchIndexEntry(entry) => validate_search_index_entry(&entry),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::SearchIndexEntry(entry) => validate_search_index_entry(&entry),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterUpdate(update_entry) => match update_entry {
            OpUpdate::Entry { app_entry, .. } => match app_entry {
                EntryTypes::SearchIndexEntry(entry) => validate_search_index_entry(&entry),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            // All link types are valid for search indexing
            match link_type {
                LinkTypes::EntityToIndex => Ok(ValidateCallbackResult::Valid),
                LinkTypes::TermToIndex => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AllIndices => Ok(ValidateCallbackResult::Valid),
                LinkTypes::EntityTypeToIndex => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate search index entry data
fn validate_search_index_entry(entry: &SearchIndexEntry) -> ExternResult<ValidateCallbackResult> {
    // Index ID required
    if entry.index_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Index ID cannot be empty".into(),
        ));
    }

    // Title required
    if entry.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }

    // Title length check
    if entry.title.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Title too long (max 500 characters)".into(),
        ));
    }

    // Description required
    if entry.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Description cannot be empty".into(),
        ));
    }

    // Description length check
    if entry.description.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 10000 characters)".into(),
        ));
    }

    // Processed terms required for searchability
    if entry.processed_terms.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Processed terms cannot be empty".into(),
        ));
    }

    // MATL score validation (0.0 - 1.0)
    if entry.creator_matl_score < 0.0 || entry.creator_matl_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator MATL score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
