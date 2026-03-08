//! Curation Coordinator Zome
use hdk::prelude::*;
use media_curation_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, requirement_for_voting,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("civic_bridge", requirement, action_name)
}

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn endorse(input: EndorseInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "endorse")?;
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    require_consciousness(&requirement_for_proposal(), "create_collection")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn feature_content(input: FeatureInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_voting(), "feature_content")?;
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    require_consciousness(&requirement_for_basic(), "add_to_collection")?;
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
                return get_latest_record(new_hash)?
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
    require_consciousness(&requirement_for_basic(), "remove_from_collection")?;
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
                return get_latest_record(action_hash)?
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
    require_consciousness(&requirement_for_proposal(), "update_collection")?;
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
                return get_latest_record(action_hash)?
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

/// Update a featured content entry
#[hdk_extern]
pub fn update_featured_content(input: UpdateFeaturedContentInput) -> ExternResult<ActionHash> {
    update_entry(
        input.original_action_hash,
        &EntryTypes::FeaturedContent(input.updated_entry),
    )
}

/// Input for updating featured content
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateFeaturedContentInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: FeaturedContent,
}

/// Update a quality score entry
#[hdk_extern]
pub fn update_quality_score(input: UpdateQualityScoreInput) -> ExternResult<ActionHash> {
    update_entry(
        input.original_action_hash,
        &EntryTypes::QualityScore(input.updated_entry),
    )
}

/// Input for updating a quality score
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateQualityScoreInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: QualityScore,
}

/// Remove endorsement
#[hdk_extern]
pub fn remove_endorsement(input: RemoveEndorsementInput) -> ExternResult<()> {
    require_consciousness(&requirement_for_basic(), "remove_endorsement")?;
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

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct RemoveEndorsementInput {
    pub endorsement_id: String,
    pub requester_did: String,
}

/// Unfeature content
#[hdk_extern]
pub fn unfeature_content(input: UnfeatureInput) -> ExternResult<()> {
    require_consciousness(&requirement_for_voting(), "unfeature_content")?;
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

/// Compute the base quality score from endorsement count.
///
/// Formula: min(endorsement_count / 100.0, 1.0)
/// Extracted as a pure function for testability.
pub fn compute_quality_base_score(endorsement_count: u32) -> f64 {
    (endorsement_count as f64 / 100.0).min(1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // QUALITY SCORE COMPUTATION TESTS
    // ========================================================================

    #[test]
    fn quality_score_zero_endorsements() {
        let score = compute_quality_base_score(0);
        assert!(
            (score - 0.0).abs() < 1e-10,
            "0 endorsements = 0.0, got {}",
            score
        );
    }

    #[test]
    fn quality_score_50_endorsements() {
        let score = compute_quality_base_score(50);
        assert!(
            (score - 0.5).abs() < 1e-10,
            "50 endorsements = 0.5, got {}",
            score
        );
    }

    #[test]
    fn quality_score_100_endorsements() {
        let score = compute_quality_base_score(100);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "100 endorsements = 1.0, got {}",
            score
        );
    }

    #[test]
    fn quality_score_200_endorsements_clamped() {
        let score = compute_quality_base_score(200);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "200 endorsements should clamp to 1.0, got {}",
            score
        );
    }

    #[test]
    fn quality_score_1_endorsement() {
        let score = compute_quality_base_score(1);
        assert!(
            (score - 0.01).abs() < 1e-10,
            "1 endorsement = 0.01, got {}",
            score
        );
    }

    #[test]
    fn quality_score_99_endorsements() {
        let score = compute_quality_base_score(99);
        assert!(
            (score - 0.99).abs() < 1e-10,
            "99 endorsements = 0.99, got {}",
            score
        );
    }

    #[test]
    fn quality_score_1000_endorsements_clamped() {
        let score = compute_quality_base_score(1000);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "1000 endorsements should clamp to 1.0, got {}",
            score
        );
    }

    #[test]
    fn quality_score_max_u32_clamped() {
        let score = compute_quality_base_score(u32::MAX);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "u32::MAX endorsements should clamp to 1.0, got {}",
            score
        );
    }

    #[test]
    fn quality_score_monotonically_increasing() {
        let mut prev = compute_quality_base_score(0);
        for i in 1..=100 {
            let current = compute_quality_base_score(i);
            assert!(
                current >= prev,
                "Score should be non-decreasing: {} < {} at i={}",
                current,
                prev,
                i
            );
            prev = current;
        }
    }

    // ========================================================================
    // INPUT STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn endorse_input_serde_roundtrip() {
        let input = EndorseInput {
            publication_id: "pub-1".into(),
            endorser_did: "did:example:endorser".into(),
            endorsement_type: EndorsementType::Upvote,
            comment: Some("Great work".into()),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: EndorseInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.publication_id, "pub-1");
        assert_eq!(parsed.endorser_did, "did:example:endorser");
        assert_eq!(parsed.comment, Some("Great work".into()));
    }

    #[test]
    fn endorse_input_no_comment_serde_roundtrip() {
        let input = EndorseInput {
            publication_id: "pub-2".into(),
            endorser_did: "did:example:e2".into(),
            endorsement_type: EndorsementType::Bookmark,
            comment: None,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: EndorseInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.comment.is_none());
    }

    #[test]
    fn create_collection_input_serde_roundtrip() {
        let input = CreateCollectionInput {
            name: "Best of 2026".into(),
            description: "Top articles of the year".into(),
            curator_did: "did:example:curator".into(),
            visibility: Visibility::Public,
            publication_ids: vec!["pub-1".into(), "pub-2".into()],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: CreateCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.name, "Best of 2026");
        assert_eq!(parsed.publication_ids.len(), 2);
    }

    #[test]
    fn feature_input_serde_roundtrip() {
        let input = FeatureInput {
            publication_id: "pub-1".into(),
            featured_by: "did:example:editor".into(),
            reason: "Outstanding reporting".into(),
            featured_until: None,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: FeatureInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.publication_id, "pub-1");
        assert!(parsed.featured_until.is_none());
    }

    #[test]
    fn add_to_collection_input_serde_roundtrip() {
        let input = AddToCollectionInput {
            collection_id: "col-1".into(),
            publication_id: "pub-3".into(),
            requester_did: "did:example:curator".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: AddToCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.collection_id, "col-1");
        assert_eq!(parsed.publication_id, "pub-3");
    }

    #[test]
    fn remove_from_collection_input_serde_roundtrip() {
        let input = RemoveFromCollectionInput {
            collection_id: "col-1".into(),
            publication_id: "pub-2".into(),
            requester_did: "did:example:curator".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: RemoveFromCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.collection_id, "col-1");
    }

    #[test]
    fn update_collection_input_serde_roundtrip() {
        let input = UpdateCollectionInput {
            collection_id: "col-1".into(),
            requester_did: "did:example:curator".into(),
            name: Some("Updated Name".into()),
            description: None,
            visibility: Some(Visibility::Private),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.name, Some("Updated Name".into()));
        assert!(parsed.description.is_none());
        assert_eq!(parsed.visibility, Some(Visibility::Private));
    }

    #[test]
    fn remove_endorsement_input_serde_roundtrip() {
        let input = RemoveEndorsementInput {
            endorsement_id: "end-1".into(),
            requester_did: "did:example:endorser".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: RemoveEndorsementInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, input);
    }

    #[test]
    fn unfeature_input_serde_roundtrip() {
        let input = UnfeatureInput {
            featured_id: "feat-1".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UnfeatureInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.featured_id, "feat-1");
    }

    // ========================================================================
    // Update input struct tests
    // ========================================================================

    #[test]
    fn update_featured_content_input_serde_roundtrip() {
        let input = UpdateFeaturedContentInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: FeaturedContent {
                id: "featured:pub-1:999".into(),
                publication_id: "pub-1".into(),
                featured_by: "did:example:editor".into(),
                reason: "Updated reason - exceptional depth".into(),
                featured_from: Timestamp::from_micros(1_000_000),
                featured_until: Some(Timestamp::from_micros(9_999_999)),
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateFeaturedContentInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(
            parsed.updated_entry.reason,
            "Updated reason - exceptional depth"
        );
        assert_eq!(
            parsed.updated_entry.featured_until,
            Some(Timestamp::from_micros(9_999_999))
        );
    }

    #[test]
    fn update_featured_content_input_clone() {
        let input = UpdateFeaturedContentInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: FeaturedContent {
                id: "feat-clone".into(),
                publication_id: "pub-c".into(),
                featured_by: "did:example:ed".into(),
                reason: "Clone test".into(),
                featured_from: Timestamp::from_micros(0),
                featured_until: None,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry.reason, "Clone test");
    }

    #[test]
    fn update_featured_content_input_no_until_serde() {
        let input = UpdateFeaturedContentInput {
            original_action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            updated_entry: FeaturedContent {
                id: "feat-no-until".into(),
                publication_id: "pub-x".into(),
                featured_by: "did:example:x".into(),
                reason: "Indefinite feature".into(),
                featured_from: Timestamp::from_micros(0),
                featured_until: None,
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateFeaturedContentInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.updated_entry.featured_until.is_none());
    }

    #[test]
    fn update_quality_score_input_serde_roundtrip() {
        let input = UpdateQualityScoreInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            updated_entry: QualityScore {
                publication_id: "pub-1".into(),
                score: 0.95,
                endorsement_count: 95,
                share_count: 50,
                fact_check_score: 0.99,
                author_reputation: 0.88,
                last_calculated: Timestamp::from_micros(5_000_000),
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateQualityScoreInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.original_action_hash,
            ActionHash::from_raw_36(vec![0xcd; 36])
        );
        assert!((parsed.updated_entry.score - 0.95).abs() < 1e-10);
        assert_eq!(parsed.updated_entry.endorsement_count, 95);
        assert_eq!(parsed.updated_entry.share_count, 50);
    }

    #[test]
    fn update_quality_score_input_clone() {
        let input = UpdateQualityScoreInput {
            original_action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            updated_entry: QualityScore {
                publication_id: "pub-clone".into(),
                score: 0.5,
                endorsement_count: 10,
                share_count: 5,
                fact_check_score: 0.5,
                author_reputation: 0.5,
                last_calculated: Timestamp::from_micros(0),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.updated_entry.publication_id, "pub-clone");
        assert!((cloned.updated_entry.score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn update_quality_score_input_zero_counts_serde() {
        let input = UpdateQualityScoreInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: QualityScore {
                publication_id: "pub-zero".into(),
                score: 0.0,
                endorsement_count: 0,
                share_count: 0,
                fact_check_score: 0.0,
                author_reputation: 0.0,
                last_calculated: Timestamp::from_micros(0),
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateQualityScoreInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.updated_entry.endorsement_count, 0);
        assert_eq!(parsed.updated_entry.share_count, 0);
        assert!((parsed.updated_entry.score - 0.0).abs() < 1e-10);
    }

    // ========================================================================
    // Additional edge-case and boundary tests
    // ========================================================================

    #[test]
    fn endorse_input_serde_award_type() {
        let input = EndorseInput {
            publication_id: "pub-award".into(),
            endorser_did: "did:example:judge".into(),
            endorsement_type: EndorsementType::Award("Best Investigative Piece".into()),
            comment: Some("Exceptional depth of research".into()),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: EndorseInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.endorsement_type,
            EndorsementType::Award("Best Investigative Piece".into())
        );
        assert_eq!(
            parsed.comment.as_deref(),
            Some("Exceptional depth of research")
        );
    }

    #[test]
    fn endorsement_type_all_variants_serde() {
        let variants = vec![
            EndorsementType::Upvote,
            EndorsementType::Bookmark,
            EndorsementType::Share,
            EndorsementType::Recommend,
            EndorsementType::Award("Gold Medal".into()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: EndorsementType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant, "Failed for {:?}", variant);
        }
    }

    #[test]
    fn create_collection_input_empty_publications() {
        let input = CreateCollectionInput {
            name: "Empty Collection".into(),
            description: "Will be filled later".into(),
            curator_did: "did:example:curator".into(),
            visibility: Visibility::Private,
            publication_ids: vec![],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: CreateCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.publication_ids.is_empty());
        assert_eq!(parsed.visibility, Visibility::Private);
    }

    #[test]
    fn update_collection_input_all_none_fields() {
        let input = UpdateCollectionInput {
            collection_id: "col-unchanged".into(),
            requester_did: "did:example:curator".into(),
            name: None,
            description: None,
            visibility: None,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateCollectionInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.name.is_none());
        assert!(parsed.description.is_none());
        assert!(parsed.visibility.is_none());
    }

    #[test]
    fn visibility_all_variants_serde() {
        let variants = vec![
            Visibility::Public,
            Visibility::Private,
            Visibility::Unlisted,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: Visibility = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn feature_input_serde_with_featured_until() {
        let input = FeatureInput {
            publication_id: "pub-featured".into(),
            featured_by: "did:example:editor".into(),
            reason: "Weekly highlight".into(),
            featured_until: Some(Timestamp::from_micros(9_999_999)),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: FeatureInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.featured_until,
            Some(Timestamp::from_micros(9_999_999))
        );
    }

    #[test]
    fn quality_score_boundary_just_below_clamp() {
        // 99 endorsements = 0.99, not yet clamped
        let score = compute_quality_base_score(99);
        assert!(score < 1.0);
        assert!((score - 0.99).abs() < 1e-10);
    }

    #[test]
    fn quality_score_boundary_exactly_at_clamp() {
        // 100 endorsements = exactly 1.0
        let score_at = compute_quality_base_score(100);
        let score_above = compute_quality_base_score(101);
        assert!((score_at - 1.0).abs() < 1e-10);
        assert!((score_above - 1.0).abs() < 1e-10);
    }
}
