// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search Integrity Zome
//!
//! Local search indexing for fast email queries.
//! Implements inverted indexes and trigram matching.

use hdi::prelude::*;

/// Search index entry (inverted index)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SearchIndex {
    /// Index ID
    pub index_id: String,
    /// Agent who owns this index
    pub agent: AgentPubKey,
    /// Index type
    pub index_type: IndexType,
    /// When created
    pub created_at: Timestamp,
    /// When last updated
    pub updated_at: Timestamp,
    /// Number of documents indexed
    pub document_count: u32,
    /// Index version for compatibility
    pub version: u32,
}

/// Types of indexes
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    /// Full-text search index
    FullText,
    /// Subject line index
    Subject,
    /// Sender index
    Sender,
    /// Recipient index
    Recipient,
    /// Label/tag index
    Label,
    /// Date range index
    DateRange,
    /// Attachment filename index
    Attachment,
}

/// Term entry in inverted index
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IndexTerm {
    /// The term (word, trigram, or value)
    pub term: String,
    /// Index type this term belongs to
    pub index_type: IndexType,
    /// Document references containing this term
    pub documents: Vec<DocumentRef>,
    /// Term frequency across all documents
    pub total_frequency: u32,
    /// Inverse document frequency (for TF-IDF)
    pub idf: f32,
}

/// Reference to a document in the index
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DocumentRef {
    /// Document hash (email hash)
    pub document_hash: ActionHash,
    /// Term frequency in this document
    pub term_frequency: u32,
    /// Positions of term in document (for phrase matching)
    pub positions: Vec<u32>,
    /// Boost factor for this document
    pub boost: f32,
}

/// Trigram for fuzzy matching
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Trigram {
    /// The three-character sequence
    pub chars: String,
    /// Terms containing this trigram
    pub terms: Vec<String>,
}

/// Search result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SearchResult {
    /// Document hash
    pub document_hash: ActionHash,
    /// Relevance score
    pub score: f32,
    /// Matched terms
    pub matched_terms: Vec<String>,
    /// Highlighted snippets
    pub snippets: Vec<String>,
    /// Document metadata
    pub metadata: DocumentMetadata,
}

/// Document metadata for search results
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DocumentMetadata {
    /// Subject line
    pub subject: String,
    /// Sender
    pub sender: AgentPubKey,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Document type
    pub doc_type: DocumentType,
    /// Preview text
    pub preview: String,
}

/// Types of indexed documents
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DocumentType {
    Email,
    Draft,
    Attachment,
    Contact,
}

/// Saved search query
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SavedSearch {
    /// Search ID
    pub id: String,
    /// Search name
    pub name: String,
    /// Query string
    pub query: String,
    /// Filters
    pub filters: SearchFilters,
    /// Owner
    pub owner: AgentPubKey,
    /// Created at
    pub created_at: Timestamp,
    /// Use count
    pub use_count: u32,
}

/// Search filters
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SearchFilters {
    /// Filter by folder
    pub folder: Option<ActionHash>,
    /// Filter by labels
    pub labels: Option<Vec<String>>,
    /// Filter by sender
    pub from: Option<AgentPubKey>,
    /// Filter by date range
    pub date_from: Option<Timestamp>,
    pub date_to: Option<Timestamp>,
    /// Has attachment
    pub has_attachment: Option<bool>,
    /// Is unread
    pub is_unread: Option<bool>,
    /// Is starred
    pub is_starred: Option<bool>,
}

/// Search statistics
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SearchStats {
    /// Total documents indexed
    pub total_documents: u32,
    /// Total unique terms
    pub total_terms: u32,
    /// Average document length
    pub avg_doc_length: f32,
    /// Last index time (ms)
    pub last_index_time_ms: u32,
    /// Last query time (ms)
    pub last_query_time_ms: u32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Link types for search index
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> search index
    AgentToIndex,
    /// Term hash -> term entry
    TermHashToTerm,
    /// Trigram hash -> trigram entry
    TrigramHashToTrigram,
    /// Agent -> saved searches
    AgentToSavedSearches,
    /// Index -> stats
    IndexToStats,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    SearchIndex(SearchIndex),
    IndexTerm(IndexTerm),
    Trigram(Trigram),
    SavedSearch(SavedSearch),
    SearchStats(SearchStats),
}

// ==================== VALIDATION ====================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::SearchIndex(index) => {
            if index.agent != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Index owner must match author".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::SavedSearch(search) => {
            if search.owner != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Search owner must match author".to_string(),
                ));
            }
            if search.name.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Search name cannot be empty".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
