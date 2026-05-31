// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Media Bridge Coordinator Zome
//!
//! Cross-hApp communication for content verification, fact-checking,
//! and author reputation across the Mycelix ecosystem.

use hdk::prelude::*;
use media_bridge_integrity::*;

const MEDIA_HAPP_ID: &str = "mycelix-media";

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

/// Verify content from another hApp
#[hdk_extern]
pub fn verify_content(input: VerifyContentInput) -> ExternResult<ContentVerificationResult> {
    let now = sys_time()?;

    let request = ContentVerificationRequest {
        id: format!(
            "verify:{}:{}:{}",
            input.source_happ,
            input.content_hash,
            now.as_micros()
        ),
        content_hash: input.content_hash.clone(),
        source_happ: input.source_happ.clone(),
        verification_type: input.verification_type,
        requested_at: now,
    };
    create_entry(&EntryTypes::ContentVerificationRequest(request))?;

    // Would perform actual verification in production
    let result = ContentVerificationResult {
        id: format!("result:{}:{}", input.content_hash, now.as_micros()),
        content_hash: input.content_hash.clone(),
        author_did: input.expected_author,
        is_verified: true,
        fact_check_score: Some(0.7),
        epistemic_classification: Some(EpistemicClassification {
            empirical: 0.6,
            normative: 0.3,
            mythic: 0.1,
        }),
        sources: vec![],
        verified_at: now,
    };

    let hash = create_entry(&EntryTypes::ContentVerificationResult(result.clone()))?;
    create_link(
        anchor_hash(&input.content_hash)?,
        hash,
        LinkTypes::ContentToVerification,
        (),
    )?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyContentInput {
    pub content_hash: String,
    pub source_happ: String,
    pub verification_type: VerificationType,
    pub expected_author: Option<String>,
}

/// Query author reputation
#[hdk_extern]
pub fn query_author_reputation(
    input: QueryAuthorReputationInput,
) -> ExternResult<AuthorReputationResult> {
    let now = sys_time()?;

    let query = AuthorReputationQuery {
        id: format!(
            "author:{}:{}:{}",
            input.source_happ,
            input.author_did,
            now.as_micros()
        ),
        author_did: input.author_did.clone(),
        source_happ: input.source_happ,
        queried_at: now,
    };
    create_entry(&EntryTypes::AuthorReputationQuery(query))?;

    // Would calculate from actual publications in production
    let result = AuthorReputationResult {
        id: format!("rep:{}:{}", input.author_did, now.as_micros()),
        author_did: input.author_did.clone(),
        publication_count: 0,
        average_quality_score: 0.5,
        fact_check_accuracy: 0.7,
        endorsement_count: 0,
        matl_score: 0.5, // Would query identity hApp
        calculated_at: now,
    };

    let hash = create_entry(&EntryTypes::AuthorReputationResult(result.clone()))?;
    create_link(
        anchor_hash(&input.author_did)?,
        hash,
        LinkTypes::AuthorToReputation,
        (),
    )?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryAuthorReputationInput {
    pub author_did: String,
    pub source_happ: String,
}

/// Register content reference from another hApp
#[hdk_extern]
pub fn register_content_reference(input: RegisterContentReferenceInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let reference = ContentReference {
        id: format!(
            "ref:{}:{}:{}",
            input.source_happ,
            input.content_hash,
            now.as_micros()
        ),
        content_hash: input.content_hash.clone(),
        source_happ: input.source_happ.clone(),
        title: input.title,
        author_did: input.author_did,
        content_type: input.content_type,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::ContentReference(reference))?;

    create_link(
        anchor_hash(&input.source_happ)?,
        hash.clone(),
        LinkTypes::HappToContent,
        (),
    )?;

    broadcast_media_event(BroadcastMediaEventInput {
        event_type: MediaEventType::ContentPublished,
        content_hash: Some(input.content_hash),
        author_did: None,
        payload: "{}".to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Reference not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterContentReferenceInput {
    pub content_hash: String,
    pub source_happ: String,
    pub title: String,
    pub author_did: String,
    pub content_type: ContentType,
}

/// Broadcast media event
#[hdk_extern]
pub fn broadcast_media_event(input: BroadcastMediaEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = MediaBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        content_hash: input.content_hash,
        author_did: input.author_did,
        payload: input.payload,
        source_happ: MEDIA_HAPP_ID.to_string(),
        timestamp: now,
    };

    let hash = create_entry(&EntryTypes::MediaBridgeEvent(event))?;

    create_link(
        anchor_hash("recent_events")?,
        hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastMediaEventInput {
    pub event_type: MediaEventType,
    pub content_hash: Option<String>,
    pub author_did: Option<String>,
    pub payload: String,
}

/// Get content from a hApp
#[hdk_extern]
pub fn get_content_by_happ(source_happ: String) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::new(
        anchor_hash(&source_happ)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::HappToContent as u8).into()),
    );
    let links = get_links(query, GetStrategy::default())?;

    let mut content = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            content.push(record);
        }
    }

    Ok(content)
}
