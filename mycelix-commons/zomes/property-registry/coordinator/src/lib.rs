// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property Registry Coordinator Zome
use commons_types::batch::links_to_records;
use hdk::prelude::*;
use mycelix_bridge_common::{
    civic_requirement_constitutional, civic_requirement_proposal, civic_requirement_voting,
};
use mycelix_zome_helpers::get_latest_record;
use property_registry_integrity::*;


/// Get or create an anchor entry and return its EntryHash for use as link base.
///
/// Anchor creation is idempotent — duplicates are expected and harmless.
/// Non-duplicate errors are logged for debugging but don't fail the operation.
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) {
        debug!("Anchor creation returned error (may be duplicate): {:?}", e);
    }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn register_property(input: RegisterPropertyInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "register_property")?;
    let now = sys_time()?;
    let property = Property {
        id: format!("property:{}:{}", input.owner_did, now.as_micros()),
        property_type: input.property_type,
        title: input.title,
        description: input.description,
        owner_did: input.owner_did.clone(),
        co_owners: input.co_owners,
        geolocation: input.geolocation.clone(),
        address: input.address,
        metadata: input.metadata,
        registered: now,
        last_transfer: None,
    };

    let action_hash = create_entry(&EntryTypes::Property(property.clone()))?;
    create_link(
        anchor_hash(&input.owner_did)?,
        action_hash.clone(),
        LinkTypes::OwnerToProperties,
        (),
    )?;

    // O(1) property ID index for direct lookup
    create_link(
        anchor_hash(&format!("property:{}", property.id))?,
        action_hash.clone(),
        LinkTypes::PropertyIdIndex,
        (),
    )?;

    // Link by location if available
    if let Some(ref geo) = input.geolocation {
        let geo_key = format!(
            "geo:{}:{}",
            (geo.latitude * 1000.0) as i64,
            (geo.longitude * 1000.0) as i64
        );
        create_link(
            anchor_hash(&geo_key)?,
            action_hash.clone(),
            LinkTypes::LocationToProperty,
            (),
        )?;

        // Geohash spatial index
        let geo_hash = commons_types::geo::geohash_encode(geo.latitude, geo.longitude, 6);
        let geo_anchor = anchor_hash(&format!("geo:{}", geo_hash))?;
        create_link(geo_anchor, action_hash.clone(), LinkTypes::GeoIndex, geo_hash.as_bytes().to_vec())?;
    }

    // Create initial title deed
    let deed = TitleDeed {
        id: format!("deed:{}:{}", property.id, now.as_micros()),
        property_id: property.id,
        owner_did: input.owner_did,
        deed_type: DeedType::Original,
        issued: now,
        previous_deed_id: None,
        encumbrances: Vec::new(),
    };
    let deed_hash = create_entry(&EntryTypes::TitleDeed(deed.clone()))?;
    create_link(
        action_hash.clone(),
        deed_hash.clone(),
        LinkTypes::PropertyToDeeds,
        (),
    )?;
    // O(1) deed lookup by property ID
    create_link(
        anchor_hash(&format!("deed:{}", deed.property_id))?,
        deed_hash,
        LinkTypes::DeedIdIndex,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterPropertyInput {
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
}

/// Get a property by ID via O(1) link-based index.
///
/// Falls back to chain scan for properties created before the index existed.
#[hdk_extern]
pub fn get_property(property_id: String) -> ExternResult<Option<Record>> {
    // O(1) path: PropertyIdIndex anchor → link → record
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("property:{}", property_id))?,
            LinkTypes::PropertyIdIndex,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid link target: {:?}", e))))?;
        return get_latest_record(action_hash);
    }

    // Fallback: chain scan for pre-index properties
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == property_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all properties owned by a DID
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_owner_properties(did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::OwnerToProperties)?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

#[hdk_extern]
pub fn add_encumbrance(input: AddEncumbranceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "add_encumbrance")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == input.property_id {
                let now = sys_time()?;
                let new_encumbrance = Encumbrance {
                    encumbrance_type: input.encumbrance_type,
                    holder_did: input.holder_did,
                    amount: input.amount,
                    registered: now,
                    expires: input.expires,
                };
                let mut encumbrances = deed.encumbrances.clone();
                encumbrances.push(new_encumbrance);
                let updated = TitleDeed {
                    encumbrances,
                    ..deed
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::TitleDeed(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddEncumbranceInput {
    pub property_id: String,
    pub encumbrance_type: EncumbranceType,
    pub holder_did: String,
    pub amount: Option<f64>,
    pub expires: Option<Timestamp>,
}

/// Search for properties by location
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn search_by_location(input: LocationSearchInput) -> ExternResult<Vec<Record>> {
    let geo_key = format!(
        "geo:{}:{}",
        (input.latitude * 1000.0) as i64,
        (input.longitude * 1000.0) as i64
    );
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&geo_key)?, LinkTypes::LocationToProperty)?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LocationSearchInput {
    pub latitude: f64,
    pub longitude: f64,
    pub radius_km: f64,
}

/// Get title deed for a property
#[hdk_extern]
pub fn get_title_deed(property_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == property_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all title deeds for a property (history)
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_property_deeds(property_id: String) -> ExternResult<Vec<Record>> {
    let property = get_property(property_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))?;

    let links = get_links(
        LinkQuery::try_new(
            property.action_address().clone(),
            LinkTypes::PropertyToDeeds,
        )?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Update property metadata
#[hdk_extern]
pub fn update_property_metadata(input: UpdateMetadataInput) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_property_metadata")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can update
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can update metadata".into()
                    )));
                }

                let updated = Property {
                    metadata: input.metadata,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMetadataInput {
    pub property_id: String,
    pub requester_did: String,
    pub metadata: PropertyMetadata,
}

/// Remove an encumbrance (when paid off)
#[hdk_extern]
pub fn remove_encumbrance(input: RemoveEncumbranceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "remove_encumbrance")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == input.property_id {
                let encumbrances: Vec<Encumbrance> = deed
                    .encumbrances
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != input.encumbrance_index)
                    .map(|(_, e)| e.clone())
                    .collect();

                let updated = TitleDeed {
                    encumbrances,
                    ..deed
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::TitleDeed(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveEncumbranceInput {
    pub property_id: String,
    pub encumbrance_index: usize,
}

/// Get properties by type
#[hdk_extern]
pub fn get_properties_by_type(property_type: PropertyType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.property_type == property_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Add a co-owner to property
#[hdk_extern]
pub fn add_co_owner(input: AddCoOwnerInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "add_co_owner")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can add co-owners
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can add co-owners".into()
                    )));
                }

                // Check total shares don't exceed 100%
                let current_shares: f64 =
                    property.co_owners.iter().map(|c| c.share_percentage).sum();
                if current_shares + input.co_owner.share_percentage > 100.0 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Total shares would exceed 100%".into()
                    )));
                }

                let mut co_owners = property.co_owners.clone();
                co_owners.push(input.co_owner);

                let updated = Property {
                    co_owners,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddCoOwnerInput {
    pub property_id: String,
    pub requester_did: String,
    pub co_owner: CoOwner,
}

/// Remove a co-owner from property
#[hdk_extern]
pub fn remove_co_owner(input: RemoveCoOwnerInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "remove_co_owner")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can remove co-owners
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can remove co-owners".into()
                    )));
                }

                let co_owners: Vec<CoOwner> = property
                    .co_owners
                    .iter()
                    .filter(|c| c.did != input.co_owner_did)
                    .cloned()
                    .collect();

                let updated = Property {
                    co_owners,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveCoOwnerInput {
    pub property_id: String,
    pub requester_did: String,
    pub co_owner_did: String,
}

/// Get all encumbrances for a property
#[hdk_extern]
pub fn get_encumbrances(property_id: String) -> ExternResult<Vec<Encumbrance>> {
    let deed = get_title_deed(property_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Title deed not found".into()
    )))?;

    let deed_data = deed
        .entry()
        .to_app_option::<TitleDeed>()
        .ok()
        .flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid deed data".into()
        )))?;

    Ok(deed_data.encumbrances)
}

/// Check if property has any active encumbrances
#[hdk_extern]
pub fn has_clear_title(property_id: String) -> ExternResult<bool> {
    let encumbrances = get_encumbrances(property_id)?;
    let now = sys_time()?;

    // Check for any active (non-expired) encumbrances
    for enc in encumbrances {
        if let Some(expires) = enc.expires {
            if expires.as_micros() > now.as_micros() {
                return Ok(false);
            }
        } else {
            // No expiry means still active
            return Ok(false);
        }
    }
    Ok(true)
}

// =============================================================================
// Ownership Transfer Functions
// =============================================================================

/// Transfer ownership of a property to a new owner.
///
/// This function:
/// 1. Updates the Property entry with new owner and transfer timestamp
/// 2. Creates a new TitleDeed with the new owner
/// 3. Links the new deed to the previous deed (deed chain)
/// 4. Carries forward any encumbrances from the previous deed
/// 5. Updates owner links
#[hdk_extern]
pub fn transfer_ownership(input: TransferOwnershipInput) -> ExternResult<TransferOwnershipResult> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_constitutional(), "transfer_ownership")?;
    let now = sys_time()?;

    // 1. Get the current property
    let property_record = get_property(input.property_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Property not found".into())
    ))?;

    let property = property_record
        .entry()
        .to_app_option::<Property>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid property data".into()
        )))?;

    // 2. Verify the current owner matches from_did
    if property.owner_did != input.from_did {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Current owner {} does not match from_did {}",
            property.owner_did, input.from_did
        ))));
    }

    // 3. Get the current title deed
    let deed_record = get_title_deed(input.property_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Title deed not found".into())
    ))?;

    let old_deed = deed_record
        .entry()
        .to_app_option::<TitleDeed>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid deed data".into()
        )))?;

    // 4. Update the Property entry with new owner
    let updated_property = Property {
        owner_did: input.to_did.clone(),
        last_transfer: Some(now),
        ..property.clone()
    };

    let property_action_hash = update_entry(
        property_record.action_address().clone(),
        &EntryTypes::Property(updated_property),
    )?;

    // 5. Remove old owner link and create new owner link
    // (Links are immutable, so we just create a new one - cleanup handled by app)
    create_link(
        anchor_hash(&input.to_did)?,
        property_action_hash.clone(),
        LinkTypes::OwnerToProperties,
        (),
    )?;

    // 6. Create new TitleDeed with new owner
    let deed_type = match input.transfer_type.as_str() {
        "Sale" => DeedType::Transfer,
        "Inheritance" => DeedType::Inheritance,
        "Gift" => DeedType::Transfer, // Could add DeedType::Gift if needed
        "CourtOrder" => DeedType::CourtOrder,
        _ => DeedType::Transfer,
    };

    let new_deed = TitleDeed {
        id: format!("deed:{}:{}", input.property_id, now.as_micros()),
        property_id: input.property_id.clone(),
        owner_did: input.to_did.clone(),
        deed_type,
        issued: now,
        previous_deed_id: Some(old_deed.id.clone()),
        encumbrances: old_deed.encumbrances.clone(), // Carry forward encumbrances
    };

    let deed_action_hash = create_entry(&EntryTypes::TitleDeed(new_deed.clone()))?;

    // 7. Link new deed to property
    create_link(
        property_action_hash.clone(),
        deed_action_hash.clone(),
        LinkTypes::PropertyToDeeds,
        (),
    )?;

    Ok(TransferOwnershipResult {
        property_action_hash,
        new_deed_id: new_deed.id,
        deed_action_hash,
        previous_deed_id: old_deed.id,
        encumbrances_carried: old_deed.encumbrances.len() as u32,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferOwnershipInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    /// Transfer type (Sale, Inheritance, Gift, CourtOrder)
    pub transfer_type: String,
    /// Optional reference to the transfer record in transfer zome
    pub transfer_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferOwnershipResult {
    pub property_action_hash: ActionHash,
    pub new_deed_id: String,
    pub deed_action_hash: ActionHash,
    pub previous_deed_id: String,
    pub encumbrances_carried: u32,
}

/// Get ownership history for a property
#[hdk_extern]
pub fn get_ownership_history(property_id: String) -> ExternResult<Vec<OwnershipRecord>> {
    let deeds = get_property_deeds(property_id)?;

    let mut history: Vec<OwnershipRecord> = Vec::new();

    for record in deeds {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            history.push(OwnershipRecord {
                deed_id: deed.id,
                owner_did: deed.owner_did,
                deed_type: deed.deed_type,
                issued: deed.issued,
                previous_deed_id: deed.previous_deed_id,
            });
        }
    }

    // Sort by issue date (oldest first)
    history.sort_by_key(|r| r.issued.as_micros());

    Ok(history)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OwnershipRecord {
    pub deed_id: String,
    pub owner_did: String,
    pub deed_type: DeedType,
    pub issued: Timestamp,
    pub previous_deed_id: Option<String>,
}

/// Verify that a DID owns a property
#[hdk_extern]
pub fn verify_ownership(input: VerifyOwnershipInput) -> ExternResult<bool> {
    let property_opt = get_property(input.property_id)?;

    if let Some(record) = property_opt {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            return Ok(property.owner_did == input.did);
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyOwnershipInput {
    pub property_id: String,
    pub did: String,
}

/// Get properties near a given location using geohash-based proximity search.
///
/// Queries the geo-index anchors created by `register_property()` to find
/// all properties within the geohash neighborhood (9 cells at precision 6,
/// approximately 3.6km x 1.8km coverage).
#[hdk_extern]
pub fn get_nearby_properties(input: commons_types::geo::NearbyQuery) -> ExternResult<Vec<Record>> {
    let center_hash = commons_types::geo::geohash_encode(input.latitude, input.longitude, 6);
    let mut all_cells = vec![center_hash.clone()];
    all_cells.extend(commons_types::geo::geohash_neighbors(&center_hash));

    let mut records = Vec::new();
    for cell in &all_cells {
        let anchor_str = format!("geo:{}", cell);
        let anchor_entry = Anchor(anchor_str);
        let anchor_hash = hash_entry(&anchor_entry)?;
        if let Ok(links) = get_links(
            LinkQuery::try_new(anchor_hash, LinkTypes::GeoIndex)?,
            GetStrategy::Local,
        ) {
            for link in links {
                if let Ok(action_hash) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(action_hash, GetOptions::default())? {
                        records.push(record);
                    }
                }
            }
        }
    }

    // Geohash precision 6 covers ~1.2km x 0.6km per cell.
    // 9 cells (center + 8 neighbors) covers ~3.6km x 1.8km.
    // For finer filtering, callers can deserialize entries and check haversine distance.
    let _ = input.radius_km; // Available for future post-filtering

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_property_input_serde_roundtrip() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Building,
            title: "Test House".to_string(),
            description: "A test property".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![CoOwner {
                did: "did:key:z6Mk002".to_string(),
                share_percentage: 25.0,
            }],
            geolocation: Some(GeoLocation {
                latitude: 32.9483,
                longitude: -96.7299,
                boundaries: None,
                area_sqm: Some(500.0),
            }),
            address: Some(Address {
                street: "123 Main St".to_string(),
                city: "Richardson".to_string(),
                region: "TX".to_string(),
                country: "US".to_string(),
                postal_code: Some("75080".to_string()),
            }),
            metadata: PropertyMetadata {
                appraised_value: Some(250_000.0),
                currency: Some("USD".to_string()),
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "Test House");
        assert_eq!(decoded.owner_did, "did:key:z6Mk001");
        assert_eq!(decoded.co_owners.len(), 1);
    }

    #[test]
    fn add_encumbrance_input_serde_roundtrip() {
        let input = AddEncumbranceInput {
            property_id: "prop-001".to_string(),
            encumbrance_type: EncumbranceType::Mortgage,
            holder_did: "did:key:z6MkBank".to_string(),
            amount: Some(150_000.0),
            expires: Some(Timestamp::from_micros(9_000_000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-001");
        assert_eq!(decoded.amount, Some(150_000.0));
    }

    #[test]
    fn location_search_input_serde_roundtrip() {
        let input = LocationSearchInput {
            latitude: 32.9483,
            longitude: -96.7299,
            radius_km: 5.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.latitude - 32.9483).abs() < f64::EPSILON);
        assert!((decoded.radius_km - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_metadata_input_serde_roundtrip() {
        let input = UpdateMetadataInput {
            property_id: "prop-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
            metadata: PropertyMetadata {
                appraised_value: Some(300_000.0),
                currency: Some("USD".to_string()),
                legal_description: Some("Lot 1".to_string()),
                parcel_number: None,
                attachments: vec!["doc.pdf".to_string()],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateMetadataInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-001");
        assert_eq!(decoded.metadata.appraised_value, Some(300_000.0));
    }

    #[test]
    fn remove_encumbrance_input_serde_roundtrip() {
        let input = RemoveEncumbranceInput {
            property_id: "prop-001".to_string(),
            encumbrance_index: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RemoveEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.encumbrance_index, 0);
    }

    #[test]
    fn add_co_owner_input_serde_roundtrip() {
        let input = AddCoOwnerInput {
            property_id: "prop-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
            co_owner: CoOwner {
                did: "did:key:z6Mk003".to_string(),
                share_percentage: 30.0,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddCoOwnerInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.co_owner.share_percentage - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn remove_co_owner_input_serde_roundtrip() {
        let input = RemoveCoOwnerInput {
            property_id: "prop-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
            co_owner_did: "did:key:z6Mk003".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RemoveCoOwnerInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.co_owner_did, "did:key:z6Mk003");
    }

    #[test]
    fn verify_ownership_input_serde_roundtrip() {
        let input = VerifyOwnershipInput {
            property_id: "prop-001".to_string(),
            did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyOwnershipInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.did, "did:key:z6Mk001");
    }

    #[test]
    fn transfer_ownership_input_serde_roundtrip() {
        let input = TransferOwnershipInput {
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6Mk001".to_string(),
            to_did: "did:key:z6Mk002".to_string(),
            transfer_type: "Sale".to_string(),
            transfer_id: Some("transfer-001".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.transfer_type, "Sale");
        assert_eq!(decoded.transfer_id, Some("transfer-001".to_string()));
    }

    #[test]
    fn transfer_ownership_result_serde_roundtrip() {
        let result = TransferOwnershipResult {
            property_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_deed_id: "deed-002".to_string(),
            deed_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            previous_deed_id: "deed-001".to_string(),
            encumbrances_carried: 1,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: TransferOwnershipResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_deed_id, "deed-002");
        assert_eq!(decoded.encumbrances_carried, 1);
    }

    #[test]
    fn ownership_record_serde_roundtrip() {
        let record = OwnershipRecord {
            deed_id: "deed-001".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            deed_type: DeedType::Original,
            issued: Timestamp::from_micros(1_000_000),
            previous_deed_id: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        let decoded: OwnershipRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.deed_id, "deed-001");
        assert_eq!(decoded.previous_deed_id, None);
    }

    #[test]
    fn register_input_without_optional_fields() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Land,
            title: "Bare Land".to_string(),
            description: "Empty lot".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.geolocation.is_none());
        assert!(decoded.address.is_none());
        assert!(decoded.co_owners.is_empty());
    }

    // ========================================================================
    // PropertyType enum variant coverage
    // ========================================================================

    #[test]
    fn property_type_all_variants_serde_roundtrip() {
        let variants = vec![
            PropertyType::Land,
            PropertyType::Building,
            PropertyType::Unit,
            PropertyType::Equipment,
            PropertyType::Intellectual,
            PropertyType::Digital,
            PropertyType::Other("CustomType".to_string()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: PropertyType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // DeedType enum variant coverage
    // ========================================================================

    #[test]
    fn deed_type_all_variants_serde_roundtrip() {
        let variants = vec![
            DeedType::Original,
            DeedType::Transfer,
            DeedType::Inheritance,
            DeedType::CourtOrder,
            DeedType::Fractional,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: DeedType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // EncumbranceType enum variant coverage
    // ========================================================================

    #[test]
    fn encumbrance_type_all_variants_serde_roundtrip() {
        let variants = vec![
            EncumbranceType::Mortgage,
            EncumbranceType::Lien,
            EncumbranceType::Easement,
            EncumbranceType::Restriction,
            EncumbranceType::Lease,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: EncumbranceType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Boundary condition tests
    // ========================================================================

    #[test]
    fn register_input_empty_title() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Building,
            title: "".to_string(),
            description: "test".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "");
    }

    #[test]
    fn register_input_empty_description() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Land,
            title: "Title".to_string(),
            description: "".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description, "");
    }

    #[test]
    fn register_input_very_long_title() {
        let long_title = "A".repeat(10_000);
        let input = RegisterPropertyInput {
            property_type: PropertyType::Digital,
            title: long_title.clone(),
            description: "test".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title.len(), 10_000);
    }

    #[test]
    fn location_search_input_zero_radius() {
        let input = LocationSearchInput {
            latitude: 0.0,
            longitude: 0.0,
            radius_km: 0.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.radius_km, 0.0);
    }

    #[test]
    fn location_search_input_extreme_coordinates() {
        let input = LocationSearchInput {
            latitude: 90.0,
            longitude: -180.0,
            radius_km: 20000.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.latitude - 90.0).abs() < f64::EPSILON);
        assert!((decoded.longitude - (-180.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn add_encumbrance_input_no_amount_no_expires() {
        let input = AddEncumbranceInput {
            property_id: "prop-002".to_string(),
            encumbrance_type: EncumbranceType::Easement,
            holder_did: "did:key:z6MkNeighbor".to_string(),
            amount: None,
            expires: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.amount.is_none());
        assert!(decoded.expires.is_none());
    }

    #[test]
    fn add_encumbrance_input_zero_amount() {
        let input = AddEncumbranceInput {
            property_id: "prop-003".to_string(),
            encumbrance_type: EncumbranceType::Restriction,
            holder_did: "did:key:z6MkGov".to_string(),
            amount: Some(0.0),
            expires: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, Some(0.0));
    }

    #[test]
    fn remove_encumbrance_input_large_index() {
        let input = RemoveEncumbranceInput {
            property_id: "prop-001".to_string(),
            encumbrance_index: usize::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RemoveEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.encumbrance_index, usize::MAX);
    }

    // ========================================================================
    // Multi-field struct field access tests
    // ========================================================================

    #[test]
    fn register_input_multiple_co_owners() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Building,
            title: "Shared House".to_string(),
            description: "A property with multiple co-owners".to_string(),
            owner_did: "did:key:z6Mk001".to_string(),
            co_owners: vec![
                CoOwner {
                    did: "did:key:z6Mk002".to_string(),
                    share_percentage: 20.0,
                },
                CoOwner {
                    did: "did:key:z6Mk003".to_string(),
                    share_percentage: 15.0,
                },
                CoOwner {
                    did: "did:key:z6Mk004".to_string(),
                    share_percentage: 10.0,
                },
            ],
            geolocation: Some(GeoLocation {
                latitude: -33.8688,
                longitude: 151.2093,
                boundaries: Some(vec![(1.0, 2.0), (3.0, 4.0)]),
                area_sqm: Some(200.0),
            }),
            address: Some(Address {
                street: "1 Opera House".to_string(),
                city: "Sydney".to_string(),
                region: "NSW".to_string(),
                country: "AU".to_string(),
                postal_code: Some("2000".to_string()),
            }),
            metadata: PropertyMetadata {
                appraised_value: Some(1_500_000.0),
                currency: Some("AUD".to_string()),
                legal_description: Some("Lot 42".to_string()),
                parcel_number: Some("P-42".to_string()),
                attachments: vec!["plan.pdf".to_string(), "survey.pdf".to_string()],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.co_owners.len(), 3);
        assert!((decoded.co_owners[0].share_percentage - 20.0).abs() < f64::EPSILON);
        assert!((decoded.co_owners[1].share_percentage - 15.0).abs() < f64::EPSILON);
        assert!((decoded.co_owners[2].share_percentage - 10.0).abs() < f64::EPSILON);
        assert_eq!(decoded.metadata.attachments.len(), 2);
        assert_eq!(
            decoded
                .geolocation
                .as_ref()
                .unwrap()
                .boundaries
                .as_ref()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn transfer_ownership_input_all_transfer_types() {
        for transfer_type in ["Sale", "Inheritance", "Gift", "CourtOrder", "Unknown"] {
            let input = TransferOwnershipInput {
                property_id: "prop-001".to_string(),
                from_did: "did:key:z6Mk001".to_string(),
                to_did: "did:key:z6Mk002".to_string(),
                transfer_type: transfer_type.to_string(),
                transfer_id: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.transfer_type, transfer_type);
        }
    }

    #[test]
    fn transfer_ownership_input_no_transfer_id() {
        let input = TransferOwnershipInput {
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6Mk001".to_string(),
            to_did: "did:key:z6Mk002".to_string(),
            transfer_type: "Sale".to_string(),
            transfer_id: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.transfer_id.is_none());
    }

    #[test]
    fn ownership_record_with_previous_deed() {
        let record = OwnershipRecord {
            deed_id: "deed-002".to_string(),
            owner_did: "did:key:z6Mk002".to_string(),
            deed_type: DeedType::Transfer,
            issued: Timestamp::from_micros(2_000_000),
            previous_deed_id: Some("deed-001".to_string()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let decoded: OwnershipRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.previous_deed_id, Some("deed-001".to_string()));
        assert_eq!(decoded.deed_type, DeedType::Transfer);
    }

    #[test]
    fn transfer_result_zero_encumbrances() {
        let result = TransferOwnershipResult {
            property_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_deed_id: "deed-new".to_string(),
            deed_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            previous_deed_id: "deed-old".to_string(),
            encumbrances_carried: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: TransferOwnershipResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.encumbrances_carried, 0);
    }

    #[test]
    fn update_metadata_input_all_none_metadata() {
        let input = UpdateMetadataInput {
            property_id: "prop-001".to_string(),
            requester_did: "did:key:z6Mk001".to_string(),
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateMetadataInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.metadata.appraised_value.is_none());
        assert!(decoded.metadata.currency.is_none());
        assert!(decoded.metadata.legal_description.is_none());
        assert!(decoded.metadata.parcel_number.is_none());
        assert!(decoded.metadata.attachments.is_empty());
    }

    #[test]
    fn co_owner_share_boundary_value_100() {
        let co = CoOwner {
            did: "did:key:z6Mk002".to_string(),
            share_percentage: 100.0,
        };
        let json = serde_json::to_string(&co).unwrap();
        let decoded: CoOwner = serde_json::from_str(&json).unwrap();
        assert!((decoded.share_percentage - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn co_owner_share_very_small() {
        let co = CoOwner {
            did: "did:key:z6Mk002".to_string(),
            share_percentage: 0.001,
        };
        let json = serde_json::to_string(&co).unwrap();
        let decoded: CoOwner = serde_json::from_str(&json).unwrap();
        assert!((decoded.share_percentage - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn address_without_postal_code() {
        let addr = Address {
            street: "Unknown Rd".to_string(),
            city: "Nowhere".to_string(),
            region: "NA".to_string(),
            country: "XX".to_string(),
            postal_code: None,
        };
        let json = serde_json::to_string(&addr).unwrap();
        let decoded: Address = serde_json::from_str(&json).unwrap();
        assert!(decoded.postal_code.is_none());
        assert_eq!(decoded.street, "Unknown Rd");
    }

    #[test]
    fn geolocation_without_boundaries_and_area() {
        let geo = GeoLocation {
            latitude: 0.0,
            longitude: 0.0,
            boundaries: None,
            area_sqm: None,
        };
        let json = serde_json::to_string(&geo).unwrap();
        let decoded: GeoLocation = serde_json::from_str(&json).unwrap();
        assert!(decoded.boundaries.is_none());
        assert!(decoded.area_sqm.is_none());
    }

    // ========================================================================
    // ERROR-PATH & FAILURE-MODE TESTS
    // ========================================================================

    /// RegisterPropertyInput with empty owner_did: verify the empty string
    /// roundtrips faithfully. The integrity layer rejects non-"did:" prefixed
    /// owner_did, but the coordinator struct must serialize it correctly.
    #[test]
    fn register_input_empty_owner_did_roundtrip() {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Land,
            title: "Test".to_string(),
            description: "Test".to_string(),
            owner_did: "".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: None,
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RegisterPropertyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.owner_did, "",
            "Empty owner_did must survive roundtrip"
        );
    }

    /// AddEncumbranceInput with empty property_id: the coordinator's
    /// add_encumbrance function searches for a deed matching the property_id
    /// and returns "Property not found" when none matches. Verify the empty
    /// string roundtrips for the error path input.
    #[test]
    fn add_encumbrance_empty_property_id_roundtrip() {
        let input = AddEncumbranceInput {
            property_id: "".to_string(),
            encumbrance_type: EncumbranceType::Lien,
            holder_did: "did:key:z6MkBank".to_string(),
            amount: Some(1000.0),
            expires: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.property_id, "",
            "Empty property_id must survive roundtrip"
        );
    }

    /// LocationSearchInput with invalid coordinates: lat > 90 and lon > 180.
    /// The integrity validation rejects these for Property entries, but the
    /// search input struct has no validation -- verify these extreme values
    /// roundtrip correctly so the search function can handle them.
    #[test]
    fn location_search_invalid_lat_over_90_roundtrip() {
        let input = LocationSearchInput {
            latitude: 91.5,
            longitude: 0.0,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert!(
            (decoded.latitude - 91.5).abs() < f64::EPSILON,
            "Lat > 90 must survive roundtrip"
        );
    }

    #[test]
    fn location_search_invalid_lon_over_180_roundtrip() {
        let input = LocationSearchInput {
            latitude: 0.0,
            longitude: 200.0,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert!(
            (decoded.longitude - 200.0).abs() < f64::EPSILON,
            "Lon > 180 must survive roundtrip"
        );
    }

    #[test]
    fn location_search_negative_radius_roundtrip() {
        let input = LocationSearchInput {
            latitude: 32.0,
            longitude: -96.0,
            radius_km: -5.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LocationSearchInput = serde_json::from_str(&json).unwrap();
        assert!(
            (decoded.radius_km - (-5.0)).abs() < f64::EPSILON,
            "Negative radius must survive roundtrip"
        );
    }

    /// GeoLocation with zero area: the integrity validation rejects area <= 0.
    /// Verify the struct roundtrips the zero value that would trigger rejection.
    #[test]
    fn geolocation_zero_area_roundtrip() {
        let geo = GeoLocation {
            latitude: 32.9,
            longitude: -96.7,
            boundaries: None,
            area_sqm: Some(0.0),
        };
        let json = serde_json::to_string(&geo).unwrap();
        let decoded: GeoLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.area_sqm,
            Some(0.0),
            "Zero area must survive roundtrip"
        );
    }

    /// GeoLocation with negative area: the integrity validation rejects this.
    /// Verify the struct roundtrips the negative value correctly.
    #[test]
    fn geolocation_negative_area_roundtrip() {
        let geo = GeoLocation {
            latitude: 32.9,
            longitude: -96.7,
            boundaries: None,
            area_sqm: Some(-500.0),
        };
        let json = serde_json::to_string(&geo).unwrap();
        let decoded: GeoLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.area_sqm,
            Some(-500.0),
            "Negative area must survive roundtrip"
        );
    }

    /// GeoLocation with extremely large area: verify f64 precision is
    /// maintained at large magnitudes.
    #[test]
    fn geolocation_very_large_area_roundtrip() {
        let large_area = 1.0e15; // 1 quadrillion sqm
        let geo = GeoLocation {
            latitude: 0.0,
            longitude: 0.0,
            boundaries: None,
            area_sqm: Some(large_area),
        };
        let json = serde_json::to_string(&geo).unwrap();
        let decoded: GeoLocation = serde_json::from_str(&json).unwrap();
        assert!(
            (decoded.area_sqm.unwrap() - large_area).abs() < 1.0,
            "Very large area must survive roundtrip"
        );
    }

    /// PropertyMetadata with negative appraised_value: the integrity
    /// validation rejects appraised_value < 0. Verify the struct roundtrips
    /// the negative value correctly for the error path.
    #[test]
    fn property_metadata_negative_appraised_value_roundtrip() {
        let meta = PropertyMetadata {
            appraised_value: Some(-100_000.0),
            currency: Some("USD".to_string()),
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let decoded: PropertyMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.appraised_value,
            Some(-100_000.0),
            "Negative value must survive roundtrip"
        );
    }

    /// TransferOwnershipInput: verify that from_did == to_did (transferring
    /// to self) roundtrips correctly. The coordinator checks ownership against
    /// from_did at runtime, but the struct must preserve self-transfer data.
    #[test]
    fn transfer_ownership_self_transfer_roundtrip() {
        let input = TransferOwnershipInput {
            property_id: "prop-001".to_string(),
            from_did: "did:key:z6Mk001".to_string(),
            to_did: "did:key:z6Mk001".to_string(),
            transfer_type: "Sale".to_string(),
            transfer_id: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.from_did, decoded.to_did,
            "Self-transfer DIDs must match after roundtrip"
        );
    }

    /// TransferOwnershipInput with empty property_id: the coordinator returns
    /// "Property not found" for this. Verify the struct roundtrips the empty
    /// string.
    #[test]
    fn transfer_ownership_empty_property_id_roundtrip() {
        let input = TransferOwnershipInput {
            property_id: "".to_string(),
            from_did: "did:key:z6Mk001".to_string(),
            to_did: "did:key:z6Mk002".to_string(),
            transfer_type: "Sale".to_string(),
            transfer_id: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferOwnershipInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.property_id, "",
            "Empty property_id must survive roundtrip"
        );
    }

    /// PropertyType::Other with empty string: verify that the Other variant
    /// can hold an empty string and roundtrip correctly.
    #[test]
    fn property_type_other_empty_string_roundtrip() {
        let pt = PropertyType::Other("".to_string());
        let json = serde_json::to_string(&pt).unwrap();
        let decoded: PropertyType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, PropertyType::Other("".to_string()));
    }

    /// Invalid PropertyType deserialization: verify that an unknown variant
    /// in JSON fails to deserialize.
    #[test]
    fn property_type_invalid_variant_deser_fails() {
        let bad_json = r#""Spacecraft""#;
        let result = serde_json::from_str::<PropertyType>(bad_json);
        assert!(
            result.is_err(),
            "Unknown PropertyType variant should fail deserialization"
        );
    }

    /// Invalid EncumbranceType deserialization: verify that an unknown variant
    /// fails to deserialize.
    #[test]
    fn encumbrance_type_invalid_variant_deser_fails() {
        let bad_json = r#""Tax""#;
        let result = serde_json::from_str::<EncumbranceType>(bad_json);
        assert!(
            result.is_err(),
            "Unknown EncumbranceType variant should fail deserialization"
        );

        let bad_json2 = r#""mortgage""#; // lowercase
        let result2 = serde_json::from_str::<EncumbranceType>(bad_json2);
        assert!(
            result2.is_err(),
            "Lowercase variant should fail deserialization"
        );
    }

    /// AddEncumbranceInput with negative amount: the integrity validation
    /// rejects negative encumbrance amounts. Verify the struct roundtrips
    /// the negative value for the error path.
    #[test]
    fn add_encumbrance_negative_amount_roundtrip() {
        let input = AddEncumbranceInput {
            property_id: "prop-001".to_string(),
            encumbrance_type: EncumbranceType::Mortgage,
            holder_did: "did:key:z6MkBank".to_string(),
            amount: Some(-50_000.0),
            expires: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEncumbranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.amount,
            Some(-50_000.0),
            "Negative encumbrance amount must survive roundtrip"
        );
    }
}
