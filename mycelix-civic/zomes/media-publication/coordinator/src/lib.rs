// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Publication Coordinator Zome
use hdk::prelude::*;
use media_publication_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::get_latest_record;


/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn publish(input: PublishInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "publish")?;

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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "add_content_block")?;

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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_publication")?;

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
                return get_latest_record(action_hash)?
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
    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in query(filter)? {
        if let Some(pub_entry) = record.entry().to_app_option::<Publication>().ok().flatten() {
            if pub_entry.id == publication_id {
                found = Some(record);
            }
        }
    }
    Ok(found)
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
                return get_latest_record(new_hash)?
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
                return get_latest_record(action_hash)?
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

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct AuthorStats {
    pub author_did: String,
    pub publication_count: u32,
    pub total_versions: u32,
}

/// Pure function: aggregate author stats from a list of version numbers.
///
/// Each entry in `versions` represents the current version of one publication.
pub fn compute_author_stats(author_did: String, versions: &[u32]) -> AuthorStats {
    let publication_count = versions.len() as u32;
    let total_versions: u32 = versions.iter().copied().fold(0u32, u32::saturating_add);
    AuthorStats {
        author_did,
        publication_count,
        total_versions,
    }
}

// ============================================================================
// Visual Art Metadata
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateArtMetadataInput {
    pub publication_hash: ActionHash,
    pub dimensions: Option<String>,
    pub medium: Option<String>,
    pub edition_size: Option<u32>,
    pub ipfs_preview_cid: Option<String>,
    pub ipfs_full_cid: Option<String>,
    pub provenance_chain: Vec<String>,
}

/// Attach visual art metadata to a publication (Participant+).
#[hdk_extern]
pub fn create_art_metadata(input: CreateArtMetadataInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "create_art_metadata")?;

    let metadata = VisualArtMetadata {
        publication_id: input.publication_hash.to_string(),
        dimensions: input.dimensions,
        medium: input.medium,
        edition_size: input.edition_size,
        ipfs_preview_cid: input.ipfs_preview_cid,
        ipfs_full_cid: input.ipfs_full_cid,
        provenance_chain: input.provenance_chain,
    };

    let action_hash = create_entry(&EntryTypes::VisualArtMetadata(metadata))?;

    // Link publication to its art metadata
    create_link(
        input.publication_hash,
        action_hash.clone(),
        LinkTypes::PublicationToMetadata,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Get art metadata for a publication.
#[hdk_extern]
pub fn get_art_metadata(publication_hash: ActionHash) -> ExternResult<Option<VisualArtMetadata>> {
    let links = get_links(
        LinkQuery::new(
            publication_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToMetadata as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(target) = link.target.clone().into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                return Ok(record.entry().to_app_option().ok().flatten());
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // compute_author_stats tests
    // ========================================================================

    #[test]
    fn author_stats_no_publications() {
        let stats = compute_author_stats("did:mycelix:alice".into(), &[]);
        assert_eq!(stats.publication_count, 0);
        assert_eq!(stats.total_versions, 0);
        assert_eq!(stats.author_did, "did:mycelix:alice");
    }

    #[test]
    fn author_stats_single_publication_v1() {
        let stats = compute_author_stats("did:mycelix:bob".into(), &[1]);
        assert_eq!(stats.publication_count, 1);
        assert_eq!(stats.total_versions, 1);
    }

    #[test]
    fn author_stats_single_publication_many_versions() {
        let stats = compute_author_stats("did:mycelix:carol".into(), &[7]);
        assert_eq!(stats.publication_count, 1);
        assert_eq!(stats.total_versions, 7);
    }

    #[test]
    fn author_stats_multiple_publications() {
        // 3 publications: v1, v3, v5 => total_versions = 9
        let stats = compute_author_stats("did:mycelix:dave".into(), &[1, 3, 5]);
        assert_eq!(stats.publication_count, 3);
        assert_eq!(stats.total_versions, 9);
    }

    #[test]
    fn author_stats_all_v1() {
        let stats = compute_author_stats("did:mycelix:eve".into(), &[1, 1, 1, 1]);
        assert_eq!(stats.publication_count, 4);
        assert_eq!(stats.total_versions, 4);
    }

    #[test]
    fn author_stats_preserves_did() {
        let stats = compute_author_stats("did:key:z6Mk_special".into(), &[2]);
        assert_eq!(stats.author_did, "did:key:z6Mk_special");
    }

    // ========================================================================
    // Serde roundtrip tests for coordinator-local structs
    // ========================================================================

    #[test]
    fn publish_input_serde_roundtrip() {
        let input = PublishInput {
            title: "My Article".into(),
            content_hash: "QmHash123".into(),
            content_type: ContentType::Article,
            author_did: "did:mycelix:alice".into(),
            co_authors: vec!["did:mycelix:bob".into()],
            language: "en".into(),
            tags: vec!["tech".into(), "science".into()],
            license: License {
                license_type: LicenseType::CCBY,
                attribution_required: true,
                commercial_use: true,
                derivative_works: true,
            },
            encrypted: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: PublishInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.title, "My Article");
        assert_eq!(input2.content_hash, "QmHash123");
        assert_eq!(input2.author_did, "did:mycelix:alice");
        assert_eq!(input2.co_authors.len(), 1);
        assert_eq!(input2.tags.len(), 2);
        assert!(!input2.encrypted);
    }

    #[test]
    fn publish_input_serde_empty_collections() {
        let input = PublishInput {
            title: "Minimal".into(),
            content_hash: "QmMinimal".into(),
            content_type: ContentType::Opinion,
            author_did: "did:mycelix:minimal".into(),
            co_authors: vec![],
            language: "es".into(),
            tags: vec![],
            license: License {
                license_type: LicenseType::CC0,
                attribution_required: false,
                commercial_use: true,
                derivative_works: true,
            },
            encrypted: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: PublishInput = serde_json::from_str(&json).unwrap();
        assert!(input2.co_authors.is_empty());
        assert!(input2.tags.is_empty());
        assert!(input2.encrypted);
    }

    #[test]
    fn add_block_input_serde_roundtrip() {
        let input = AddBlockInput {
            publication_id: "pub-1".into(),
            block_index: 0,
            content: "Hello world".into(),
            encrypted_content: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddBlockInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-1");
        assert_eq!(input2.block_index, 0);
        assert_eq!(input2.content, "Hello world");
        assert!(input2.encrypted_content.is_none());
    }

    #[test]
    fn add_block_input_serde_with_encrypted() {
        let input = AddBlockInput {
            publication_id: "pub-2".into(),
            block_index: 5,
            content: "cleartext".into(),
            encrypted_content: Some("enc_data_abc".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddBlockInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.encrypted_content.as_deref(), Some("enc_data_abc"));
    }

    #[test]
    fn update_input_serde_roundtrip() {
        let input = UpdateInput {
            publication_id: "pub-1".into(),
            new_content_hash: "QmNewHash".into(),
            change_summary: "Fixed typos".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: UpdateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-1");
        assert_eq!(input2.new_content_hash, "QmNewHash");
        assert_eq!(input2.change_summary, "Fixed typos");
    }

    #[test]
    fn delete_publication_input_serde_roundtrip() {
        let input = DeletePublicationInput {
            publication_id: "pub-99".into(),
            requester_did: "did:mycelix:admin".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: DeletePublicationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-99");
        assert_eq!(input2.requester_did, "did:mycelix:admin");
    }

    #[test]
    fn add_tags_input_serde_roundtrip() {
        let input = AddTagsInput {
            publication_id: "pub-1".into(),
            requester_did: "did:mycelix:alice".into(),
            new_tags: vec!["newTag1".into(), "newTag2".into()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddTagsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.new_tags.len(), 2);
        assert_eq!(input2.new_tags[0], "newTag1");
    }

    #[test]
    fn update_license_input_serde_roundtrip() {
        let input = UpdateLicenseInput {
            publication_id: "pub-1".into(),
            requester_did: "did:mycelix:alice".into(),
            new_license: License {
                license_type: LicenseType::CCBYNCSA,
                attribution_required: true,
                commercial_use: false,
                derivative_works: true,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: UpdateLicenseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-1");
        assert!(!input2.new_license.commercial_use);
        assert!(input2.new_license.derivative_works);
    }

    #[test]
    fn author_stats_serde_roundtrip() {
        let stats = AuthorStats {
            author_did: "did:mycelix:alice".into(),
            publication_count: 10,
            total_versions: 25,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let stats2: AuthorStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, stats2);
    }

    // ========================================================================
    // Additional edge-case and boundary tests
    // ========================================================================

    #[test]
    fn author_stats_large_version_numbers_no_overflow() {
        // Use large values that don't overflow u32 when summed
        let stats = compute_author_stats("did:mycelix:heavy".into(), &[u32::MAX / 2, u32::MAX / 2]);
        assert_eq!(stats.publication_count, 2);
        assert_eq!(stats.total_versions, (u32::MAX / 2) * 2);
    }

    #[test]
    fn author_stats_many_publications() {
        let versions: Vec<u32> = (1..=50).collect();
        let stats = compute_author_stats("did:mycelix:prolific".into(), &versions);
        assert_eq!(stats.publication_count, 50);
        // sum of 1..=50 = 1275
        assert_eq!(stats.total_versions, 1275);
    }

    #[test]
    fn publish_input_serde_unicode_title() {
        let input = PublishInput {
            title: "Articulo en espanol con acentos".into(),
            content_hash: "QmUnicode".into(),
            content_type: ContentType::Other("ensayo".into()),
            author_did: "did:mycelix:unicode_author".into(),
            co_authors: vec![],
            language: "es".into(),
            tags: vec!["ciencia".into()],
            license: License {
                license_type: LicenseType::CC0,
                attribution_required: false,
                commercial_use: true,
                derivative_works: true,
            },
            encrypted: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: PublishInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.title, "Articulo en espanol con acentos");
        assert_eq!(input2.language, "es");
    }

    #[test]
    fn publish_input_serde_many_co_authors() {
        let co_authors: Vec<String> = (0..20)
            .map(|i| format!("did:mycelix:coauthor{}", i))
            .collect();
        let input = PublishInput {
            title: "Collaborative Work".into(),
            content_hash: "QmCollab".into(),
            content_type: ContentType::Report,
            author_did: "did:mycelix:lead".into(),
            co_authors: co_authors.clone(),
            language: "en".into(),
            tags: vec![],
            license: License {
                license_type: LicenseType::CCBYSA,
                attribution_required: true,
                commercial_use: true,
                derivative_works: true,
            },
            encrypted: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: PublishInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.co_authors.len(), 20);
        assert_eq!(input2.co_authors[0], "did:mycelix:coauthor0");
        assert_eq!(input2.co_authors[19], "did:mycelix:coauthor19");
    }

    #[test]
    fn add_block_input_serde_max_block_index() {
        let input = AddBlockInput {
            publication_id: "pub-boundary".into(),
            block_index: u32::MAX,
            content: "Last block".into(),
            encrypted_content: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddBlockInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.block_index, u32::MAX);
    }

    #[test]
    fn add_tags_input_serde_empty_tags() {
        let input = AddTagsInput {
            publication_id: "pub-1".into(),
            requester_did: "did:mycelix:alice".into(),
            new_tags: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddTagsInput = serde_json::from_str(&json).unwrap();
        assert!(input2.new_tags.is_empty());
    }

    #[test]
    fn update_license_input_serde_custom_license() {
        let input = UpdateLicenseInput {
            publication_id: "pub-custom".into(),
            requester_did: "did:mycelix:owner".into(),
            new_license: License {
                license_type: LicenseType::Custom("Proprietary-v2".into()),
                attribution_required: false,
                commercial_use: false,
                derivative_works: false,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: UpdateLicenseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            input2.new_license.license_type,
            LicenseType::Custom("Proprietary-v2".into())
        );
        assert!(!input2.new_license.attribution_required);
        assert!(!input2.new_license.commercial_use);
        assert!(!input2.new_license.derivative_works);
    }

    #[test]
    fn author_stats_equality() {
        let a = AuthorStats {
            author_did: "did:mycelix:test".into(),
            publication_count: 5,
            total_versions: 10,
        };
        let b = AuthorStats {
            author_did: "did:mycelix:test".into(),
            publication_count: 5,
            total_versions: 10,
        };
        assert_eq!(a, b);

        let c = AuthorStats {
            author_did: "did:mycelix:other".into(),
            publication_count: 5,
            total_versions: 10,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn publish_input_serde_all_content_types() {
        let content_types = vec![
            ContentType::Article,
            ContentType::Opinion,
            ContentType::Investigation,
            ContentType::Review,
            ContentType::Analysis,
            ContentType::Interview,
            ContentType::Report,
            ContentType::Editorial,
            ContentType::Other("podcast_transcript".into()),
        ];
        for ct in content_types {
            let input = PublishInput {
                title: "Test".into(),
                content_hash: "QmTest".into(),
                content_type: ct.clone(),
                author_did: "did:mycelix:tester".into(),
                co_authors: vec![],
                language: "en".into(),
                tags: vec![],
                license: License {
                    license_type: LicenseType::CC0,
                    attribution_required: false,
                    commercial_use: true,
                    derivative_works: true,
                },
                encrypted: false,
            };
            let json = serde_json::to_string(&input).unwrap();
            let input2: PublishInput = serde_json::from_str(&json).unwrap();
            assert_eq!(input2.content_type, ct, "Failed for content type {:?}", ct);
        }
    }
}
