// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Media Bridge Integrity Zome
//!
//! Entry types for cross-hApp content verification, fact-checking,
//! and author reputation integration.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Content verification request from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContentVerificationRequest {
    pub id: String,
    pub content_hash: String,
    pub source_happ: String,
    pub verification_type: VerificationType,
    pub requested_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationType {
    AuthorVerification,
    FactCheck,
    SourceCheck,
    PlagiarismCheck,
}

/// Content verification result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContentVerificationResult {
    pub id: String,
    pub content_hash: String,
    pub author_did: Option<String>,
    pub is_verified: bool,
    pub fact_check_score: Option<f64>,
    pub epistemic_classification: Option<EpistemicClassification>,
    pub sources: Vec<String>,
    pub verified_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicClassification {
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
}

/// Author reputation query
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AuthorReputationQuery {
    pub id: String,
    pub author_did: String,
    pub source_happ: String,
    pub queried_at: Timestamp,
}

/// Author reputation result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AuthorReputationResult {
    pub id: String,
    pub author_did: String,
    pub publication_count: u32,
    pub average_quality_score: f64,
    pub fact_check_accuracy: f64,
    pub endorsement_count: u32,
    pub matl_score: f64,
    pub calculated_at: Timestamp,
}

/// Cross-hApp content reference
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContentReference {
    pub id: String,
    pub content_hash: String,
    pub source_happ: String,
    pub title: String,
    pub author_did: String,
    pub content_type: ContentType,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContentType {
    Article,
    Report,
    Analysis,
    Opinion,
    Review,
    Evidence,
    Claim,
}

/// Media event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MediaBridgeEvent {
    pub id: String,
    pub event_type: MediaEventType,
    pub content_hash: Option<String>,
    pub author_did: Option<String>,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MediaEventType {
    ContentPublished,
    ContentVerified,
    FactCheckCompleted,
    AuthorVerified,
    ContentDisputed,
    ContentRetracted,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    ContentVerificationRequest(ContentVerificationRequest),
    ContentVerificationResult(ContentVerificationResult),
    AuthorReputationQuery(AuthorReputationQuery),
    AuthorReputationResult(AuthorReputationResult),
    ContentReference(ContentReference),
    MediaBridgeEvent(MediaBridgeEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    ContentToVerification,
    AuthorToReputation,
    HappToContent,
    RecentEvents,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ContentVerificationRequest(request) => {
                    validate_create_content_verification_request(
                        EntryCreationAction::Create(action),
                        request,
                    )
                }
                EntryTypes::ContentVerificationResult(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::AuthorReputationQuery(query) => {
                    validate_create_author_reputation_query(
                        EntryCreationAction::Create(action),
                        query,
                    )
                }
                EntryTypes::AuthorReputationResult(result) => {
                    validate_create_author_reputation_result(
                        EntryCreationAction::Create(action),
                        result,
                    )
                }
                EntryTypes::ContentReference(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MediaBridgeEvent(event) => {
                    validate_create_media_bridge_event(EntryCreationAction::Create(action), event)
                }
            },
            OpEntry::UpdateEntry { .. } => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::ContentToVerification => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AuthorToReputation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToContent => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_content_verification_request(
    _action: EntryCreationAction,
    request: ContentVerificationRequest,
) -> ExternResult<ValidateCallbackResult> {
    if request.content_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_author_reputation_query(
    _action: EntryCreationAction,
    query: AuthorReputationQuery,
) -> ExternResult<ValidateCallbackResult> {
    if !query.author_did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must have valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_author_reputation_result(
    _action: EntryCreationAction,
    result: AuthorReputationResult,
) -> ExternResult<ValidateCallbackResult> {
    if result.average_quality_score < 0.0 || result.average_quality_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quality score must be 0.0-1.0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_media_bridge_event(
    _action: EntryCreationAction,
    event: MediaBridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
