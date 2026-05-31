// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Curation Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Endorsement {
    pub id: String,
    pub publication_id: String,
    pub endorser_did: String,
    pub endorsement_type: EndorsementType,
    pub comment: Option<String>,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EndorsementType {
    Upvote,
    Bookmark,
    Share,
    Recommend,
    Award(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub visibility: Visibility,
    pub publication_ids: Vec<String>,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Unlisted,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QualityScore {
    pub publication_id: String,
    pub score: f64,
    pub endorsement_count: u32,
    pub share_count: u32,
    pub fact_check_score: f64,
    pub author_reputation: f64,
    pub last_calculated: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FeaturedContent {
    pub id: String,
    pub publication_id: String,
    pub featured_by: String,
    pub reason: String,
    pub featured_from: Timestamp,
    pub featured_until: Option<Timestamp>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Endorsement(Endorsement),
    Collection(Collection),
    QualityScore(QualityScore),
    FeaturedContent(FeaturedContent),
}

#[hdk_link_types]
pub enum LinkTypes {
    PublicationToEndorsements,
    EndorserToEndorsements,
    CuratorToCollections,
    CollectionToPublications,
    PublicationToQuality,
    FeaturedPublications,
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
                EntryTypes::Endorsement(endorsement) => {
                    validate_create_endorsement(EntryCreationAction::Create(action), endorsement)
                }
                EntryTypes::Collection(collection) => {
                    validate_create_collection(EntryCreationAction::Create(action), collection)
                }
                EntryTypes::QualityScore(score) => {
                    validate_create_quality_score(EntryCreationAction::Create(action), score)
                }
                EntryTypes::FeaturedContent(featured) => {
                    validate_create_featured_content(EntryCreationAction::Create(action), featured)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Endorsement(_) => Ok(ValidateCallbackResult::Invalid(
                    "Endorsements cannot be updated".into(),
                )),
                EntryTypes::Collection(collection) => {
                    validate_update_collection(action, collection)
                }
                EntryTypes::QualityScore(score) => validate_update_quality_score(action, score),
                EntryTypes::FeaturedContent(featured) => {
                    validate_update_featured_content(action, featured)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::PublicationToEndorsements => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EndorserToEndorsements => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CuratorToCollections => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CollectionToPublications => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicationToQuality => Ok(ValidateCallbackResult::Valid),
            LinkTypes::FeaturedPublications => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_endorsement(
    _action: EntryCreationAction,
    endorsement: Endorsement,
) -> ExternResult<ValidateCallbackResult> {
    if !endorsement.endorser_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Endorser must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_collection(
    _action: EntryCreationAction,
    collection: Collection,
) -> ExternResult<ValidateCallbackResult> {
    if !collection.curator_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Curator must be a valid DID".into(),
        ));
    }
    if collection.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_collection(
    _action: Update,
    collection: Collection,
) -> ExternResult<ValidateCallbackResult> {
    if collection.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection name required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_quality_score(
    _action: EntryCreationAction,
    score: QualityScore,
) -> ExternResult<ValidateCallbackResult> {
    if score.score < 0.0 || score.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Score must be 0-1".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_quality_score(
    _action: Update,
    score: QualityScore,
) -> ExternResult<ValidateCallbackResult> {
    if score.score < 0.0 || score.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Score must be 0-1".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_featured_content(
    _action: EntryCreationAction,
    featured: FeaturedContent,
) -> ExternResult<ValidateCallbackResult> {
    if !featured.featured_by.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Featured by must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_featured_content(
    _action: Update,
    _featured: FeaturedContent,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
