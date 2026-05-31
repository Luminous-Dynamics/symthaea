// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Publication Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Publication {
    pub id: String,
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
    pub published: Timestamp,
    pub updated: Option<Timestamp>,
    pub version: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContentType {
    Article,
    Opinion,
    Investigation,
    Review,
    Analysis,
    Interview,
    Report,
    Editorial,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct License {
    pub license_type: LicenseType,
    pub attribution_required: bool,
    pub commercial_use: bool,
    pub derivative_works: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LicenseType {
    CC0,
    CCBY,
    CCBYSA,
    CCBYNC,
    CCBYNCSA,
    AllRightsReserved,
    Custom(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContentBlock {
    pub publication_id: String,
    pub block_index: u32,
    pub content: String,
    pub encrypted_content: Option<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PublicationVersion {
    pub publication_id: String,
    pub version: u32,
    pub content_hash: String,
    pub change_summary: String,
    pub created: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Publication(Publication),
    ContentBlock(ContentBlock),
    PublicationVersion(PublicationVersion),
}

#[hdk_link_types]
pub enum LinkTypes {
    AuthorToPublications,
    TagToPublications,
    PublicationToBlocks,
    PublicationToVersions,
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
                EntryTypes::Publication(publication) => {
                    validate_create_publication(EntryCreationAction::Create(action), publication)
                }
                EntryTypes::ContentBlock(block) => {
                    validate_create_content_block(EntryCreationAction::Create(action), block)
                }
                EntryTypes::PublicationVersion(version) => validate_create_publication_version(
                    EntryCreationAction::Create(action),
                    version,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Publication(publication) => {
                    validate_update_publication(action, publication)
                }
                EntryTypes::ContentBlock(_) => Ok(ValidateCallbackResult::Invalid(
                    "Content blocks cannot be updated".into(),
                )),
                EntryTypes::PublicationVersion(_) => Ok(ValidateCallbackResult::Invalid(
                    "Publication versions cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AuthorToPublications => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TagToPublications => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicationToBlocks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicationToVersions => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_publication(
    _action: EntryCreationAction,
    publication: Publication,
) -> ExternResult<ValidateCallbackResult> {
    if !publication.author_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must be a valid DID".into(),
        ));
    }
    if publication.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }
    if publication.content_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_publication(
    _action: Update,
    publication: Publication,
) -> ExternResult<ValidateCallbackResult> {
    if publication.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_content_block(
    _action: EntryCreationAction,
    _block: ContentBlock,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_publication_version(
    _action: EntryCreationAction,
    _version: PublicationVersion,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
