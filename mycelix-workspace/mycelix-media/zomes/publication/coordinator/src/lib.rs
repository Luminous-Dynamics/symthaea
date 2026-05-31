// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Publication Coordinator Zome
use hdk::prelude::*;
use publication_integrity::*;

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn publish(input: PublishInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let publication = Publication {
        id: format!("pub:{}:{}", input.author_did, now.as_micros()),
        title: input.title,
        content_hash: input.content_hash,
        content_type: input.content_type,
        author_did: input.author_did.clone(),
        co_authors: input.co_authors,
        language: input.language,
        tags: input.tags.clone(),
        license: input.license,
        encrypted: input.encrypted,
        published: now,
        updated: None,
        version: 1,
    };

    let action_hash = create_entry(&EntryTypes::Publication(publication.clone()))?;
    create_link(
        anchor_hash(&input.author_did)?,
        action_hash.clone(),
        LinkTypes::AuthorToPublications,
        (),
    )?;

    for tag in input.tags {
        create_link(
            anchor_hash(&tag.to_lowercase())?,
            action_hash.clone(),
            LinkTypes::TagToPublications,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PublishInput {
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
}

#[hdk_extern]
pub fn add_content_block(input: AddBlockInput) -> ExternResult<Record> {
    let block = ContentBlock {
        publication_id: input.publication_id.clone(),
        block_index: input.block_index,
        content: input.content,
        encrypted_content: input.encrypted_content,
    };

    let action_hash = create_entry(&EntryTypes::ContentBlock(block))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToBlocks,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddBlockInput {
    pub publication_id: String,
    pub block_index: u32,
    pub content: String,
    pub encrypted_content: Option<String>,
}

#[hdk_extern]
pub fn update_publication(input: UpdateInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == input.publication_id {
                let now = sys_time()?;
                let new_version = pub_entry.version + 1;

                // Create version record
                let version = PublicationVersion {
                    publication_id: input.publication_id.clone(),
                    version: new_version,
                    content_hash: input.new_content_hash.clone(),
                    change_summary: input.change_summary,
                    created: now,
                };
                let version_hash = create_entry(&EntryTypes::PublicationVersion(version))?;
                create_link(
                    anchor_hash(&input.publication_id)?,
                    version_hash,
                    LinkTypes::PublicationToVersions,
                    (),
                )?;

                // Update publication
                let updated = Publication {
                    content_hash: input.new_content_hash,
                    updated: Some(now),
                    version: new_version,
                    ..pub_entry
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Publication(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Publication not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateInput {
    pub publication_id: String,
    pub new_content_hash: String,
    pub change_summary: String,
}

#[hdk_extern]
pub fn get_publication(publication_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == publication_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

#[hdk_extern]
pub fn search_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let mut publications = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&tag.to_lowercase())?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::TagToPublications as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            publications.push(record);
        }
    }
    Ok(publications)
}

/// Get all publications by author
#[hdk_extern]
pub fn get_author_publications(author_did: String) -> ExternResult<Vec<Record>> {
    let mut publications = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&author_did)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::AuthorToPublications as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            publications.push(record);
        }
    }
    Ok(publications)
}

/// Get publication content blocks
#[hdk_extern]
pub fn get_publication_blocks(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut blocks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToBlocks as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            blocks.push(record);
        }
    }
    Ok(blocks)
}

/// Get publication version history
#[hdk_extern]
pub fn get_publication_versions(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut versions = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToVersions as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            versions.push(record);
        }
    }
    Ok(versions)
}

/// Get publications by content type
#[hdk_extern]
pub fn get_publications_by_type(content_type: ContentType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.content_type == content_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Get publications by language
#[hdk_extern]
pub fn get_publications_by_language(language: String) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.language == language {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Delete a publication (soft delete by updating)
#[hdk_extern]
pub fn delete_publication(input: DeletePublicationInput) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == input.publication_id {
                // Only author can delete
                if pub_entry.author_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only author can delete".into()
                    )));
                }
                delete_entry(record.action_address().clone())?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Publication not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeletePublicationInput {
    pub publication_id: String,
    pub requester_did: String,
}

/// Add tags to existing publication
#[hdk_extern]
pub fn add_tags(input: AddTagsInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == input.publication_id {
                // Only author can add tags
                if pub_entry.author_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only author can add tags".into()
                    )));
                }

                let now = sys_time()?;
                let action_hash = record.action_address().clone();

                // Create links for new tags
                for tag in &input.new_tags {
                    create_link(
                        anchor_hash(&tag.to_lowercase())?,
                        action_hash.clone(),
                        LinkTypes::TagToPublications,
                        (),
                    )?;
                }

                let mut all_tags = pub_entry.tags.clone();
                all_tags.extend(input.new_tags);

                let updated = Publication {
                    tags: all_tags,
                    updated: Some(now),
                    ..pub_entry
                };
                let new_hash = update_entry(action_hash, &EntryTypes::Publication(updated))?;
                return get(new_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Publication not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddTagsInput {
    pub publication_id: String,
    pub requester_did: String,
    pub new_tags: Vec<String>,
}

/// Update publication license
#[hdk_extern]
pub fn update_license(input: UpdateLicenseInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == input.publication_id {
                if pub_entry.author_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only author can update license".into()
                    )));
                }

                let now = sys_time()?;
                let updated = Publication {
                    license: input.new_license,
                    updated: Some(now),
                    ..pub_entry
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Publication(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Publication not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateLicenseInput {
    pub publication_id: String,
    pub requester_did: String,
    pub new_license: License,
}

/// Get author stats
#[hdk_extern]
pub fn get_author_stats(author_did: String) -> ExternResult<AuthorStats> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Publication,
        )?))
        .include_entries(true);

    let mut publication_count = 0;
    let mut total_versions = 0;

    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.author_did == author_did {
                publication_count += 1;
                total_versions += pub_entry.version;
            }
        }
    }

    Ok(AuthorStats {
        author_did,
        publication_count,
        total_versions,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AuthorStats {
    pub author_did: String,
    pub publication_count: u32,
    pub total_versions: u32,
}
