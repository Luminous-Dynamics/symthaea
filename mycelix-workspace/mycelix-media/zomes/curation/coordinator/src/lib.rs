// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Curation Coordinator Zome
use curation_integrity::*;
use hdk::prelude::*;

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn endorse(input: EndorseInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let endorsement = Endorsement {
        id: format!(
            "endorse:{}:{}:{}",
            input.publication_id,
            input.endorser_did,
            now.as_micros()
        ),
        publication_id: input.publication_id.clone(),
        endorser_did: input.endorser_did.clone(),
        endorsement_type: input.endorsement_type,
        comment: input.comment,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::Endorsement(endorsement))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToEndorsements,
        (),
    )?;
    create_link(
        anchor_hash(&input.endorser_did)?,
        action_hash.clone(),
        LinkTypes::EndorserToEndorsements,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EndorseInput {
    pub publication_id: String,
    pub endorser_did: String,
    pub endorsement_type: EndorsementType,
    pub comment: Option<String>,
}

#[hdk_extern]
pub fn create_collection(input: CreateCollectionInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let collection = Collection {
        id: format!("collection:{}:{}", input.curator_did, now.as_micros()),
        name: input.name,
        description: input.description,
        curator_did: input.curator_did.clone(),
        visibility: input.visibility,
        publication_ids: input.publication_ids.clone(),
        created: now,
        updated: now,
    };

    let action_hash = create_entry(&EntryTypes::Collection(collection))?;
    create_link(
        anchor_hash(&input.curator_did)?,
        action_hash.clone(),
        LinkTypes::CuratorToCollections,
        (),
    )?;

    for pub_id in input.publication_ids {
        create_link(
            action_hash.clone(),
            anchor_hash(&pub_id)?,
            LinkTypes::CollectionToPublications,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCollectionInput {
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub visibility: Visibility,
    pub publication_ids: Vec<String>,
}

#[hdk_extern]
pub fn calculate_quality_score(publication_id: String) -> ExternResult<Record> {
    let now = sys_time()?;

    // Count endorsements
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::PublicationToEndorsements as u8).into(),
        ),
    );
    let endorsement_count = get_links(query, GetStrategy::default())?.len() as u32;

    // Simplified scoring
    let base_score = (endorsement_count as f64 / 100.0).min(1.0);

    let score = QualityScore {
        publication_id: publication_id.clone(),
        score: base_score,
        endorsement_count,
        share_count: 0,
        fact_check_score: 0.5,
        author_reputation: 0.5,
        last_calculated: now,
    };

    let action_hash = create_entry(&EntryTypes::QualityScore(score))?;
    create_link(
        anchor_hash(&publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToQuality,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn feature_content(input: FeatureInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let featured = FeaturedContent {
        id: format!("featured:{}:{}", input.publication_id, now.as_micros()),
        publication_id: input.publication_id.clone(),
        featured_by: input.featured_by,
        reason: input.reason,
        featured_from: now,
        featured_until: input.featured_until,
    };

    let action_hash = create_entry(&EntryTypes::FeaturedContent(featured))?;
    create_link(
        anchor_hash("featured_content")?,
        action_hash.clone(),
        LinkTypes::FeaturedPublications,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureInput {
    pub publication_id: String,
    pub featured_by: String,
    pub reason: String,
    pub featured_until: Option<Timestamp>,
}

#[hdk_extern]
pub fn get_featured_content(_: ()) -> ExternResult<Vec<Record>> {
    let mut featured = Vec::new();
    let query = LinkQuery::new(
        anchor_hash("featured_content")?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::FeaturedPublications as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            featured.push(record);
        }
    }
    Ok(featured)
}

#[hdk_extern]
pub fn get_publication_endorsements(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut endorsements = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::PublicationToEndorsements as u8).into(),
        ),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            endorsements.push(record);
        }
    }
    Ok(endorsements)
}

/// Get endorser's history
#[hdk_extern]
pub fn get_endorser_history(endorser_did: String) -> ExternResult<Vec<Record>> {
    let mut endorsements = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&endorser_did)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::EndorserToEndorsements as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            endorsements.push(record);
        }
    }
    Ok(endorsements)
}

/// Get curator's collections
#[hdk_extern]
pub fn get_curator_collections(curator_did: String) -> ExternResult<Vec<Record>> {
    let mut collections = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&curator_did)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::CuratorToCollections as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            collections.push(record);
        }
    }
    Ok(collections)
}

/// Get collection by ID
#[hdk_extern]
pub fn get_collection(collection_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Collection,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collection) = record.entry().to_app_option::<Collection>().ok().flatten() {
            if collection.id == collection_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Add publication to collection
#[hdk_extern]
pub fn add_to_collection(input: AddToCollectionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Collection,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collection) = record.entry().to_app_option::<Collection>().ok().flatten() {
            if collection.id == input.collection_id {
                // Only curator can add to collection
                if collection.curator_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only curator can modify collection".into()
                    )));
                }

                let now = sys_time()?;
                let action_hash = record.action_address().clone();

                // Create link to publication
                create_link(
                    action_hash.clone(),
                    anchor_hash(&input.publication_id)?,
                    LinkTypes::CollectionToPublications,
                    (),
                )?;

                let mut publication_ids = collection.publication_ids.clone();
                publication_ids.push(input.publication_id);

                let updated = Collection {
                    publication_ids,
                    updated: now,
                    ..collection
                };
                let new_hash = update_entry(action_hash, &EntryTypes::Collection(updated))?;
                return get(new_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Collection not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddToCollectionInput {
    pub collection_id: String,
    pub publication_id: String,
    pub requester_did: String,
}

/// Remove publication from collection
#[hdk_extern]
pub fn remove_from_collection(input: RemoveFromCollectionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Collection,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collection) = record.entry().to_app_option::<Collection>().ok().flatten() {
            if collection.id == input.collection_id {
                if collection.curator_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only curator can modify collection".into()
                    )));
                }

                let now = sys_time()?;
                let publication_ids: Vec<String> = collection
                    .publication_ids
                    .iter()
                    .filter(|id| **id != input.publication_id)
                    .cloned()
                    .collect();

                let updated = Collection {
                    publication_ids,
                    updated: now,
                    ..collection
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Collection(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Collection not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveFromCollectionInput {
    pub collection_id: String,
    pub publication_id: String,
    pub requester_did: String,
}

/// Update collection metadata
#[hdk_extern]
pub fn update_collection(input: UpdateCollectionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Collection,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collection) = record.entry().to_app_option::<Collection>().ok().flatten() {
            if collection.id == input.collection_id {
                if collection.curator_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only curator can update collection".into()
                    )));
                }

                let now = sys_time()?;
                let updated = Collection {
                    name: input.name.unwrap_or(collection.name),
                    description: input.description.unwrap_or(collection.description),
                    visibility: input.visibility.unwrap_or(collection.visibility),
                    updated: now,
                    ..collection
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Collection(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Collection not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCollectionInput {
    pub collection_id: String,
    pub requester_did: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub visibility: Option<Visibility>,
}

/// Remove endorsement
#[hdk_extern]
pub fn remove_endorsement(input: RemoveEndorsementInput) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Endorsement,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(endorsement) = record.entry().to_app_option::<Endorsement>().ok().flatten() {
            if endorsement.id == input.endorsement_id {
                // Only endorser can remove
                if endorsement.endorser_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only endorser can remove endorsement".into()
                    )));
                }
                delete_entry(record.action_address().clone())?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Endorsement not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveEndorsementInput {
    pub endorsement_id: String,
    pub requester_did: String,
}

/// Unfeature content
#[hdk_extern]
pub fn unfeature_content(input: UnfeatureInput) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FeaturedContent,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(featured) = record
            .entry()
            .to_app_option::<FeaturedContent>()
            .ok()
            .flatten()
        {
            if featured.id == input.featured_id {
                delete_entry(record.action_address().clone())?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Featured content not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UnfeatureInput {
    pub featured_id: String,
}

/// Get publication quality score
#[hdk_extern]
pub fn get_quality_score(publication_id: String) -> ExternResult<Option<Record>> {
    let mut scores = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToQuality as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            scores.push(record);
        }
    }
    Ok(scores.into_iter().last())
}

/// Get public collections
#[hdk_extern]
pub fn get_public_collections(_: ()) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Collection,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(collection) = record.entry().to_app_option::<Collection>().ok().flatten() {
            if collection.visibility == Visibility::Public {
                results.push(record);
            }
        }
    }
    Ok(results)
}
