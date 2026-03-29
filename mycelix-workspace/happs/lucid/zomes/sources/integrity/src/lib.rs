// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Sources Integrity Zome
//!
//! Source attribution and provenance tracking for knowledge units.
//!
//! Every piece of knowledge can be traced to its origin:
//! - Books, papers, articles
//! - Web pages, videos, podcasts
//! - Personal experiences, conversations
//! - Other people's claims

use hdi::prelude::*;
use sha2::{Sha256, Digest};

// ============================================================================
// ANCHOR
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// SOURCE TYPES
// ============================================================================

/// Type of source
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SourceType {
    /// Academic paper (peer-reviewed)
    AcademicPaper,
    /// Book or ebook
    Book,
    /// News article
    NewsArticle,
    /// Blog post
    BlogPost,
    /// Web page (generic)
    WebPage,
    /// Video (YouTube, lecture, etc.)
    Video,
    /// Podcast or audio
    Audio,
    /// Social media post
    SocialMedia,
    /// Personal conversation
    Conversation,
    /// Personal experience or observation
    PersonalExperience,
    /// Official document (government, legal, etc.)
    OfficialDocument,
    /// Dataset or statistics
    Dataset,
    /// Another LUCID thought
    LucidThought,
    /// External knowledge graph or database
    KnowledgeBase,
    /// Unknown or unspecified
    Other,
}

impl Default for SourceType {
    fn default() -> Self {
        SourceType::Other
    }
}

impl SourceType {
    /// Quality weight for confidence calculation (0.0-1.0)
    /// Academic sources get higher weight than social media
    pub fn quality_weight(&self) -> f64 {
        match self {
            SourceType::AcademicPaper => 1.0,
            SourceType::OfficialDocument => 0.95,
            SourceType::Dataset => 0.9,
            SourceType::Book => 0.85,
            SourceType::KnowledgeBase => 0.8,
            SourceType::NewsArticle => 0.6,
            SourceType::Video => 0.5,
            SourceType::Audio => 0.5,
            SourceType::BlogPost => 0.4,
            SourceType::WebPage => 0.35,
            SourceType::LucidThought => 0.5,
            SourceType::Conversation => 0.3,
            SourceType::PersonalExperience => 0.3,
            SourceType::SocialMedia => 0.2,
            SourceType::Other => 0.25,
        }
    }
}

// ============================================================================
// SOURCE ENTRY
// ============================================================================

/// A source of knowledge
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Source {
    /// Unique identifier
    pub id: String,

    /// Type of source
    pub source_type: SourceType,

    /// Title or name
    pub title: String,

    /// URL if available
    pub url: Option<String>,

    /// Author(s)
    pub authors: Vec<String>,

    /// Publication date
    pub publication_date: Option<String>,

    /// Publisher or platform
    pub publisher: Option<String>,

    /// DOI for academic papers
    pub doi: Option<String>,

    /// ISBN for books
    pub isbn: Option<String>,

    /// Short description or abstract
    pub description: Option<String>,

    /// SHA-256 hash of content for verification
    pub content_hash: Option<String>,

    /// Credibility score (0.0-1.0)
    pub credibility: f64,

    /// Personal notes about this source
    pub notes: Option<String>,

    /// Tags for organization
    pub tags: Vec<String>,

    /// When this source was added to LUCID
    pub added_at: Timestamp,

    /// Last time metadata was updated
    pub updated_at: Timestamp,
}

// ============================================================================
// SOURCE CITATION
// ============================================================================

/// A citation linking a thought to a source with context
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Citation {
    /// Unique identifier
    pub id: String,

    /// The thought this citation belongs to
    pub thought_id: String,

    /// The source being cited
    pub source_id: String,

    /// Specific location (page, timestamp, section)
    pub location: Option<String>,

    /// Quoted text from the source
    pub quote: Option<String>,

    /// How this source supports the thought
    pub relationship: CitationRelationship,

    /// How strongly does this source support/refute
    pub strength: f64,

    /// Personal notes on this citation
    pub notes: Option<String>,

    /// When this citation was created
    pub created_at: Timestamp,
}

/// How the source relates to the thought
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CitationRelationship {
    /// Source directly supports the claim
    Supports,
    /// Source provides evidence against
    Refutes,
    /// Source provides background context
    Context,
    /// Source is where the idea originated
    Origin,
    /// Source offers an alternative view
    Alternative,
    /// Source provides methodology
    Methodology,
    /// Source offers a definition
    Definition,
    /// Generic reference
    Reference,
}

impl Default for CitationRelationship {
    fn default() -> Self {
        CitationRelationship::Supports
    }
}

// ============================================================================
// SOURCE VERIFICATION
// ============================================================================

/// Verification of a source by another user
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SourceVerification {
    /// Source being verified
    pub source_id: String,

    /// Verifier's agent pubkey (as string)
    pub verifier: String,

    /// Verification outcome
    pub verification_type: VerificationType,

    /// Confidence in this verification
    pub confidence: f64,

    /// Comments on verification
    pub comments: Option<String>,

    /// When verified
    pub verified_at: Timestamp,
}

/// Type of verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationType {
    /// Confirmed the source exists and is accessible
    Exists,
    /// Confirmed the content matches what's claimed
    ContentVerified,
    /// Source is no longer accessible
    Inaccessible,
    /// Content has changed since citation
    ContentChanged,
    /// Source appears to be fabricated
    Fabricated,
}

// ============================================================================
// ENTRY TYPES & LINKS
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Source(Source),
    Citation(Citation),
    SourceVerification(SourceVerification),
}

#[hdk_link_types]
pub enum LinkTypes {
    // Agent links
    AgentToSource,

    // Source discovery
    SourceIdToSource,
    SourceTypeToSource,
    DomainToSource,
    AuthorToSource,

    // Citation links
    ThoughtToCitation,
    CitationToSource,
    SourceToCitation,

    // Verification
    SourceToVerification,
    VerifierToVerification,

    // URL index
    UrlToSource,
    DoiToSource,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Source(source) => validate_create_source(action, source),
                EntryTypes::Citation(citation) => validate_create_citation(action, citation),
                EntryTypes::SourceVerification(v) => validate_create_verification(action, v),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Source(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Citation(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SourceVerification(_) => Ok(ValidateCallbackResult::Invalid(
                    "Verifications cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(op_update) => {
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_create_source(_action: Create, source: Source) -> ExternResult<ValidateCallbackResult> {
    if source.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source title cannot be empty".into(),
        ));
    }

    if source.credibility < 0.0 || source.credibility > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credibility must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate URL format if provided
    if let Some(ref url) = source.url {
        if !url.starts_with("http://") && !url.starts_with("https://") && !url.starts_with("file://") {
            return Ok(ValidateCallbackResult::Invalid(
                "URL must start with http://, https://, or file://".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_citation(_action: Create, citation: Citation) -> ExternResult<ValidateCallbackResult> {
    if citation.thought_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Citation thought_id cannot be empty".into(),
        ));
    }

    if citation.source_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Citation source_id cannot be empty".into(),
        ));
    }

    if citation.strength < 0.0 || citation.strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Citation strength must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_verification(
    _action: Create,
    verification: SourceVerification,
) -> ExternResult<ValidateCallbackResult> {
    if verification.source_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Verification source_id cannot be empty".into(),
        ));
    }

    if verification.confidence < 0.0 || verification.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Verification confidence must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Compute SHA-256 hash of content
pub fn compute_content_hash(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    let result = hasher.finalize();
    hex::encode(result)
}

// Simple hex encoding (no external crate needed for basic use)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}
